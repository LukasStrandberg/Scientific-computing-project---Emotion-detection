"""Make the "3 frames with predicted emotions" image for one or more videos.

For each video, three moments are taken at 25 %, 50 % and 75 % of its length. Around each
moment, 16 frames spread over a few seconds (--span, default 3 s, similar to a MAFW clip)
(less for videos shorter than 3 x span, so the moments never overlap)
are face-cropped exactly like the training data, and the S2D model predicts the emotions
for that stretch. The frame at the moment itself is drawn with the face box, next to a bar
chart of the model's emotion scores.

Modes (--mode):
    mafw  11-class MAFW model (softmax: the probabilities add up to 1, includes "neutral")
    er    6-output model fine-tuned on ElderReact (each emotion is its own yes/no
          probability between 0 and 1; they do not add up to 1; there is no "neutral")

Examples (from the repo root, venv-s2d active):
    python s2d/annotate_video.py --mode mafw ^
        --ckpt external/S2D/ckpts/MAFW/without_oversampler/split01_checkpoint-best.pth ^
        --video "data/AlzheimerPatient_Data/Img 6705.mp4" --out results/annotated_mafw

    python s2d/annotate_video.py --mode er --ckpt results/er_ft/best.pth ^
        --video data/AlzheimerPatient_Data/*.mp4 --out results/annotated_er

Outputs in --out, per video:
    <video>_3frames.png   the image (3 rows: frame with face box | emotion scores)
    <video>_3frames.csv   moment, time, and the score of every emotion
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ER_CLASSES, MAFW_CLASSES, MEAN, STD, build_model  # noqa: E402
from preprocess_faces import align_crop, fill_missing, largest_face  # noqa: E402

BAR = "#2a78d6"
BOX = "#1baf7a"
TEXT = "#2b2b29"
MUTED = "#6b6a64"


def read_frames(cap, indices):
    out = {}
    for i in sorted(set(indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            out[i] = frame
    return out


def fmt_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", nargs="+", required=True, help="video file(s); wildcards allowed")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="mafw", choices=["mafw", "er"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--moments", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--preset", default="s2d", choices=["s2d", "maedfer"],
                    help="s2d: 3 s of video around each moment; maedfer: 1 s (like the MAE-DFER track)")
    ap.add_argument("--span", type=float, help="seconds of video around each moment (overrides --preset)")
    ap.add_argument("--top", type=int, default=6, help="how many emotions to show per moment")
    ap.add_argument("--style", default="deepface", choices=["square", "deepface"])
    args = ap.parse_args()
    if args.span is None:
        args.span = 3.0 if args.preset == "s2d" else 1.0

    videos = [Path(p) for v in args.video for p in (sorted(glob.glob(v)) or [v])]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = ER_CLASSES if args.mode == "er" else MAFW_CLASSES
    model, landmark = build_model(args.ckpt, len(names), device)
    from batch_face import RetinaFace
    detector = RetinaFace(gpu_id=0 if device == "cuda" else -1, network="resnet50", fp16=device == "cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if n <= 0:
            print(f"{vid}: cannot read, skipped")
            continue
        rows = []
        # short videos: shrink the span so the stretches around the moments do not overlap
        span = min(args.span, n / fps / len(args.moments))
        for m in args.moments:
            c = int(round(m * (n - 1)))
            half = span * fps / 2
            idx = [int(round(v)) for v in np.linspace(max(c - half, 0), min(c + half, n - 1), 16)]
            frames = read_frames(cap, idx + [c])
            idx = [i for i in idx if i in frames]
            if c not in frames or not idx:
                print(f"{vid.name} @ {m:.0%}: could not read frames, skipped")
                continue
            imgs = [frames[i] for i in idx] + [frames[c]]
            dets = [largest_face(r, 0.8) for r in detector(imgs, cv=True, threshold=0.8, max_size=640)]
            centre_det = dets[-1]
            dets = fill_missing(dets[:-1])
            if dets is None:
                print(f"{vid.name} @ {m:.0%}: no face found, skipped")
                continue
            while len(dets) < 16:  # if a few frames could not be read
                dets.append(dets[-1]); idx.append(idx[-1])
            crops = [cv2.cvtColor(align_crop(frames[i], b, k, 224, 0.0, args.style), cv2.COLOR_BGR2RGB)
                     for i, (b, k) in zip(idx, dets)]
            x = (np.stack(crops).astype(np.float32) / 255.0 - MEAN) / STD  # T H W C
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(3, 0, 1, 2)))[None].to(device)
            with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
                logits, _ = model(x, landmark(x))
            logits = logits.float()[0]
            p = (logits.sigmoid() if args.mode == "er" else logits.softmax(-1)).cpu().numpy()
            box = centre_det[0] if centre_det is not None else dets[len(dets) // 2][0]
            rows.append(dict(moment=m, frame=c, time=c / fps, probs=p, image=frames[c], box=box,
                             face_at_moment=centre_det is not None))
        cap.release()
        if not rows:
            continue

        # ---- figure: one row per moment, frame | scores
        fig, axes = plt.subplots(len(rows), 2, figsize=(10, 3.6 * len(rows)),
                                 gridspec_kw=dict(width_ratios=[1.15, 1]), squeeze=False)
        for (ax_img, ax_bar), r in zip(axes, rows):
            img = cv2.cvtColor(r["image"], cv2.COLOR_BGR2RGB)
            ax_img.imshow(img)
            x1, y1, x2, y2 = r["box"]
            ax_img.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=BOX, lw=2))
            ax_img.set_axis_off()
            note = "" if r["face_at_moment"] else "  (face box from a nearby frame)"
            ax_img.set_title(f"{fmt_time(r['time'])}  ({r['moment']:.0%} of video){note}",
                             fontsize=10, color=TEXT, loc="left")
            order = np.argsort(r["probs"])[::-1][:args.top]
            labels = [names[i] for i in order][::-1]
            vals = r["probs"][order][::-1]
            bars = ax_bar.barh(labels, vals, color=BAR, height=0.6)
            for b, v in zip(bars, vals):
                ax_bar.text(b.get_width() + 0.015, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                            va="center", fontsize=9, color=TEXT)
            ax_bar.set_xlim(0, 1.08)
            ax_bar.tick_params(colors=MUTED, labelsize=9)
            ax_bar.tick_params(axis="y", colors=TEXT, labelsize=10, length=0)
            for s in ("top", "right", "left"):
                ax_bar.spines[s].set_visible(False)
            ax_bar.spines["bottom"].set_color("#d3d2cc")
            ax_bar.set_title("predicted: " + names[order[0]], fontsize=11, color=TEXT, loc="left")
        what = ("probability (all emotions sum to 1)" if args.mode == "mafw"
                else "probability per emotion (independent yes/no, does not sum to 1)")
        axes[-1][1].set_xlabel(what, fontsize=9, color=MUTED)
        fig.suptitle(f"{vid.name}  ·  S2D ({'MAFW, 11 emotions' if args.mode == 'mafw' else 'fine-tuned on ElderReact, 6 emotions'})",
                     fontsize=12, color=TEXT, x=0.01, ha="left")
        fig.tight_layout()
        png = out_dir / f"{vid.stem}_3frames.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        with open(out_dir / f"{vid.stem}_3frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["moment", "frame", "time_s", "predicted"] + names)
            for r in rows:
                w.writerow([r["moment"], r["frame"], f"{r['time']:.2f}", names[int(np.argmax(r["probs"]))]]
                           + [f"{v:.4f}" for v in r["probs"]])
        print(f"{vid.name}: saved {png}")


if __name__ == "__main__":
    main()
