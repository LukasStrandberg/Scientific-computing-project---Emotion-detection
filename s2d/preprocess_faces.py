"""Cut aligned 224x224 face crops out of video clips, in the folder format S2D expects.

For every video, one folder of frames is written:
    <out_dir>/<clip_id>/00001.jpg, 00002.jpg, ...
plus <out_dir>/_report.csv with frame counts and how many frames had no face.

Default crop style is "deepface" (tight face box padded with black), which scored
higher than a square crop with the released S2D MAFW model.

Face detection: RetinaFace (ResNet-50), the same detector family the S2D authors used
(via deepface), here through the `batch-face` package so it runs batched on the GPU.
Per frame we keep the LARGEST face, rotate the frame so the eyes are level, take a
square box around the face and resize it to 224x224. If a frame has no face, the box
from the nearest frame that had one is reused; if a whole clip has no face, the clip
is listed in the report and skipped.

Examples (run from the repo root, with venv-s2d active):
    # smoke test: first 50 clips of the MAFW fold-1 test set
    python s2d/preprocess_faces.py --videos data/MFAW_Data/clips --out data/processed/MAFW/faces_224 --only-csv data/processed/MAFW/anno/set_1/test.csv --limit 50
    # all ElderReact splits
    python s2d/preprocess_faces.py --videos data/ElderReact_Data/ElderReact_train data/ElderReact_Data/ElderReact_dev data/ElderReact_Data/ElderReact_test --out data/processed/ElderReact/faces_224
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from batch_face import RetinaFace

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)  # BGR
    cap.release()
    return frames


def largest_face(faces, min_score):
    """faces: list of (box[x1,y1,x2,y2], kps[5x2], score). Returns (box, kps) or None."""
    faces = [f for f in faces if f[2] >= min_score]
    if not faces:
        return None
    box, kps, _ = max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
    return np.asarray(box, dtype=np.float32), np.asarray(kps, dtype=np.float32)


def fill_missing(dets):
    """Replace None entries with the nearest detected face (in time)."""
    have = [i for i, d in enumerate(dets) if d is not None]
    if not have:
        return None
    out = []
    for i, d in enumerate(dets):
        if d is None:
            j = min(have, key=lambda k: abs(k - i))
            d = dets[j]
        out.append(d)
    return out


def align_crop(frame, box, kps, size, margin, style="square"):
    """Rotate so the eyes are level, then cut out the face and resize to size x size.

    style="square":   square region of the image around the face (some background left/right)
    style="deepface": tight face box, resized keeping its aspect ratio and padded with black
                      to a square - mimics deepface's extract_faces, which the S2D authors used
    """
    (lx, ly), (rx, ry) = kps[0], kps[1]  # left eye, right eye (image coordinates)
    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = (box[2] - box[0]) * (1 + margin), (box[3] - box[1]) * (1 + margin)
    if style == "deepface":
        scale = size / max(w, h)
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, scale)
        M[0, 2] += size / 2 - cx
        M[1, 2] += size / 2 - cy
        out = cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # black out everything outside the (rotated-frame) face box
        pw, ph = int(round(w * scale)), int(round(h * scale))
        mask = np.zeros((size, size), dtype=bool)
        x0, y0 = (size - pw) // 2, (size - ph) // 2
        mask[max(y0, 0):y0 + ph, max(x0, 0):x0 + pw] = True
        out[~mask] = 0
        return out
    side = max(w, h)
    # Rotate around the face centre and shift the face centre to the middle of the output.
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, size / side)
    M[0, 2] += size / 2 - cx
    M[1, 2] += size / 2 - cy
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True, help="folder(s) with video files")
    ap.add_argument("--out", required=True, help="output folder (one subfolder per clip)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N videos (smoke test)")
    ap.add_argument("--only-csv", help="only process clips listed in this label CSV (first column = clip id)")
    ap.add_argument("--sample", type=int, default=0,
                    help="process a random sample of N videos (fixed seed, so reruns pick the same clips)")
    ap.add_argument("--style", default="deepface", choices=["square", "deepface"], help="crop style, see align_crop()")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--margin", type=float, default=0.0, help="extra border around the face box, e.g. 0.1 = 10%%")
    ap.add_argument("--min-score", type=float, default=0.8, help="minimum detection confidence")
    ap.add_argument("--batch", type=int, default=32, help="frames per detector batch")
    ap.add_argument("--max-size", type=int, default=640, help="frames are downscaled to this long side for detection")
    ap.add_argument("--network", default="resnet50", choices=["resnet50", "mobilenet"])
    ap.add_argument("--cpu", action="store_true", help="force CPU (slow)")
    ap.add_argument("--overwrite", action="store_true", help="redo clips that already have frames")
    args = ap.parse_args()

    use_gpu = torch.cuda.is_available() and not args.cpu
    detector = RetinaFace(gpu_id=0 if use_gpu else -1, network=args.network, fp16=use_gpu)
    print(f"Detector: RetinaFace-{args.network} on {'GPU' if use_gpu else 'CPU'}")

    videos = sorted(p for d in args.videos for p in Path(d).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if args.only_csv:
        with open(args.only_csv) as f:
            wanted = [line.split(",")[0].strip() for line in f.readlines()[1:] if line.strip()]
        by_stem = {v.stem: v for v in videos}
        videos = [by_stem[c] for c in wanted if c in by_stem]
    if args.sample and args.sample < len(videos):
        videos = sorted(random.Random(0).sample(videos, args.sample))
    if args.limit:
        videos = videos[: args.limit]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "_report.csv"
    new_report = not report_path.exists()

    t0 = time.time()
    with open(report_path, "a", newline="") as rf:
        rep = csv.writer(rf)
        if new_report:
            rep.writerow(["clip", "frames", "frames_without_face", "status"])
        for n, vid in enumerate(videos, 1):
            clip_dir = out_root / vid.stem
            if clip_dir.exists() and any(clip_dir.iterdir()) and not args.overwrite:
                continue
            frames = read_frames(vid)
            if not frames:
                rep.writerow([vid.stem, 0, 0, "unreadable"])
                continue
            results = detector(frames, cv=True, threshold=args.min_score,
                               max_size=args.max_size, batch_size=args.batch)
            dets = [largest_face(r, args.min_score) for r in results]
            missing = sum(d is None for d in dets)
            dets = fill_missing(dets)
            if dets is None:
                rep.writerow([vid.stem, len(frames), missing, "no_face"])
                print(f"[{n}/{len(videos)}] {vid.name}: no face found, skipped")
                continue
            clip_dir.mkdir(parents=True, exist_ok=True)
            for i, (frame, (box, kps)) in enumerate(zip(frames, dets), 1):
                crop = align_crop(frame, box, kps, args.size, args.margin, args.style)
                cv2.imwrite(str(clip_dir / f"{i:05d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rep.writerow([vid.stem, len(frames), missing, "ok"])
            rf.flush()
            if n % 50 == 0 or n == len(videos):
                el = time.time() - t0
                print(f"[{n}/{len(videos)}] {el/60:.1f} min elapsed, ~{el/n*(len(videos)-n)/60:.1f} min left")
    print(f"Done. Crops in {out_root}, report in {report_path}")


if __name__ == "__main__":
    main()
