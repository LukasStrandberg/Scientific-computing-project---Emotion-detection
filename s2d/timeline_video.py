"""Emotion-over-time timeline for long videos (the Alzheimer patient videos).

The model was trained on short clips (a few seconds) and always looks at 16 frames, so a
long video is cut into overlapping windows: every --stride seconds a window of --window
seconds is taken, 16 frames spread over it are face-cropped exactly like the training data
(largest face, deepface-style crop), and each model predicts the emotions for that window.

Frames are read on a fixed grid (one frame every window/16 seconds), each grid frame is
face-cropped once, and a window is simply 16 consecutive grid frames - so overlapping
windows share their crops and the whole video is processed in one pass. The crops are kept
in a temporary file on disk (not in memory), so long videos and short windows are fine.

Window presets (--preset), individual options override them:
    s2d      3 s windows every 1 s, smoothed over 5 windows   (matches the 1-5 s training clips)
    maedfer  1 s windows every 0.8 s, no smoothing            (same as the MAE-DFER track)

Models (one or both):
    --er-ckpt    6-output model fine-tuned on ElderReact (independent yes/no probabilities,
                 no "neutral"). Dominant emotion = highest score if it is >= --er-min,
                 otherwise "no clear emotion".
    --mafw-ckpt  11-class MAFW model (probabilities sum to 1, includes "neutral").
                 Dominant emotion = highest probability.

Outputs in --out, per video:
    <video>_timeline.png       one panel per model: smoothed scores over time (the 6 shared
                               emotions, plus neutral for MAFW) and a band with the dominant
                               emotion; times without a face are shaded
    <video>_timeline.csv       one row per window: times, share of frames with a face, raw
                               scores of every model
    <video>_segments.csv       per model: stretches where the (smoothed) dominant emotion stays
                               the same for at least --min-segment seconds = where expressions change

Example (from the repo root, venv-s2d active):
    python s2d/timeline_video.py --video "data/AlzheimerPatient_Data/*.mp4" ^
        --er-ckpt results/er_ft_short/best.pth ^
        --mafw-ckpt external/S2D/ckpts/MAFW/without_oversampler/split01_checkpoint-best.pth ^
        --out results/timeline
"""
import argparse
import csv
import glob
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ER_CLASSES, MAFW_CLASSES, MEAN, STD, build_model  # noqa: E402
from plot_tsne import STYLE  # noqa: E402  (same colour per emotion as in the t-SNE plot)
from preprocess_faces import align_crop, largest_face  # noqa: E402

TEXT, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3dd"
NO_FACE = "#f0efea"
NONE_COLOR = "#d3d2cc"   # "no clear emotion"
RARE_COLOR = "#6b6a64"   # MAFW's 4 rare emotions (anxiety, contempt, disappointment, helplessness)
RARE_MAFW = ["anxiety", "contempt", "disappointment", "helplessness"]
PRESETS = {"s2d": dict(window=3.0, stride=1.0, smooth=5),
           "maedfer": dict(window=1.0, stride=0.8, smooth=1)}
SHOWN_MAFW = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]


def fmt_time(seconds, _pos=None):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def crop_grid(video, step_s, detector, style, crop_file, batch=32):
    """Read one frame every step_s seconds. Returns (times, crops, has_face, duration);
    crops is a uint8 RGB array (N, 224, 224, 3) stored in crop_file on disk."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = sorted({int(round(t * fps)) for t in np.arange(0, n / fps, step_s)} - {n})
    wanted_set, times, frames = set(wanted), [], []
    crops = np.lib.format.open_memmap(crop_file, mode="w+", dtype=np.uint8, shape=(len(wanted), 224, 224, 3))
    count, has_face, last = 0, [], None
    t0 = time.time()

    def flush():
        nonlocal last, count
        dets = detector(frames, cv=True, threshold=0.8, max_size=640, batch_size=batch)
        for fr, r in zip(frames, dets):
            d = largest_face(r, 0.8)
            has_face.append(d is not None)
            if d is not None:
                last = d
            if last is not None:  # before the first face: stays black, marked as no-face
                crops[count] = cv2.cvtColor(align_crop(fr, last[0], last[1], 224, 0.0, style), cv2.COLOR_BGR2RGB)
            count += 1
        frames.clear()

    i = 0
    while i < n:
        if i in wanted_set:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr); times.append(i / fps)
            if len(frames) == batch:
                flush()
                if count % (batch * 20) == 0:
                    print(f"    {times[-1] / 60:.1f} / {n / fps / 60:.1f} min of video  ({time.time() - t0:.0f} s)")
        else:
            if not cap.grab():
                break
        i += 1
    if frames:
        flush()
    cap.release()
    return np.array(times), crops[:count], np.array(has_face), n / fps


def remove_tmp(path):
    """Delete the temporary crop file (call after all references to the array are gone)."""
    import gc
    gc.collect()
    try:
        os.remove(path)
    except OSError as e:
        print(f"  (could not delete temporary file {path}: {e})")


@torch.no_grad()
def run_model(model, landmark, crops, starts, device, sigmoid, batch=8):
    out = []
    for b in range(0, len(starts), batch):
        clips = [np.stack(crops[s:s + 16]) for s in starts[b:b + batch]]
        x = (np.stack(clips).astype(np.float32) / 255.0 - MEAN) / STD  # B T H W C
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))).to(device)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            logits, _ = model(x, landmark(x))
        logits = logits.float()
        out.append((logits.sigmoid() if sigmoid else logits.softmax(-1)).cpu().numpy())
    return np.concatenate(out)


def smooth(x, k):
    """Centred moving average over k windows (ignores NaN)."""
    if k <= 1:
        return x
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - k // 2): i + k // 2 + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmean(seg, axis=0)
    return out


def dominant(scores, names, min_score):
    lab = []
    for row in scores:
        if not np.isfinite(row).all():
            lab.append("no face")
        elif row.max() < min_score:
            lab.append("no clear emotion")
        else:
            lab.append(names[int(np.argmax(row))])
    return lab


def segments(labels, centres, stride, min_len):
    """Merge runs of equal labels; runs shorter than min_len seconds are absorbed."""
    runs = []
    for lab, t in zip(labels, centres):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = t
        else:
            runs.append([lab, t, t])
    changed = True
    while changed and len(runs) > 1:  # absorb short runs into the longer neighbour
        changed = False
        for i, r in enumerate(runs):
            if r[2] - r[1] + stride < min_len:
                if i == 0:
                    j = 1
                elif i == len(runs) - 1:
                    j = i - 1
                else:
                    j = i - 1 if (runs[i - 1][2] - runs[i - 1][1]) >= (runs[i + 1][2] - runs[i + 1][1]) else i + 1
                lo, hi = min(i, j), max(i, j)
                runs[lo] = [runs[j][0], runs[lo][1], runs[hi][2]]
                del runs[hi]
                changed = True
                break
        # merge neighbours that became equal
        k = 1
        while k < len(runs):
            if runs[k][0] == runs[k - 1][0]:
                runs[k - 1][2] = runs[k][2]
                del runs[k]
            else:
                k += 1
    return [(lab, max(a - stride / 2, 0), b + stride / 2) for lab, a, b in runs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", nargs="+", required=True, help="video file(s); wildcards allowed")
    ap.add_argument("--er-ckpt", help="ElderReact fine-tuned checkpoint (6 outputs)")
    ap.add_argument("--mafw-ckpt", help="MAFW checkpoint (11 classes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="s2d", choices=list(PRESETS), help="window settings, see above")
    ap.add_argument("--window", type=float, help="seconds per window (16 frames spread over it)")
    ap.add_argument("--stride", type=float, help="seconds between window starts")
    ap.add_argument("--smooth", type=int, help="moving average over this many windows for plot/segments (1 = none)")
    ap.add_argument("--er-min", type=float, default=0.5,
                    help="ElderReact model: below this top score the window counts as 'no clear emotion'")
    ap.add_argument("--min-face", type=float, default=0.5, help="windows with a smaller share of face frames = no face")
    ap.add_argument("--min-segment", type=float, default=5.0, help="seconds; shorter stretches are merged away")
    ap.add_argument("--style", default="deepface", choices=["square", "deepface"])
    ap.add_argument("--tmp-dir", help="folder for the temporary crop file (default: system temp folder)")
    args = ap.parse_args()
    for key, val in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    print(f"Windows: {args.window:g} s every {args.stride:g} s, smoothing {args.smooth} (preset '{args.preset}')")
    if not (args.er_ckpt or args.mafw_ckpt):
        sys.exit("Give --er-ckpt and/or --mafw-ckpt")

    videos = [Path(p) for v in args.video for p in (sorted(glob.glob(v)) or [v])]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU (slow)")
    models = []  # (key, title, names, sigmoid, model, landmark, min_score)
    if args.er_ckpt:
        m, lm = build_model(args.er_ckpt, 6, device)
        models.append(("er", "S2D fine-tuned on ElderReact (6 emotions, independent probabilities)",
                       ER_CLASSES, True, m, lm, args.er_min))
    if args.mafw_ckpt:
        m, lm = build_model(args.mafw_ckpt, 11, device)
        models.append(("mafw", "S2D trained on MAFW (11 emotions, probabilities sum to 1)",
                       MAFW_CLASSES, False, m, lm, 0.0))
    from batch_face import RetinaFace
    detector = RetinaFace(gpu_id=0 if device == "cuda" else -1, network="resnet50", fp16=device == "cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = args.window / 16
    k = max(int(round(args.stride / step)), 1)
    for vid in videos:
        print(f"{vid.name}: face crops every {step:.3f} s ...")
        t0 = time.time()
        tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=args.tmp_dir)
        tmp.close()
        times, crops, has_face, duration = crop_grid(vid, step, detector, args.style, tmp.name)
        if len(crops) < 16:
            print(f"{vid.name}: shorter than one window, skipped")
            del crops
            remove_tmp(tmp.name)
            continue
        starts = list(range(0, len(crops) - 15, k))
        centres = np.array([(times[s] + times[s + 15]) / 2 for s in starts])
        face_share = np.array([has_face[s:s + 16].mean() for s in starts])
        valid = face_share >= args.min_face
        print(f"  {len(crops)} frames cropped ({100 * has_face.mean():.1f} % with a face), "
              f"{len(starts)} windows ({time.time() - t0:.0f} s)")

        results = {}
        for key, title, names, sig, m, lm, min_score in models:
            t1 = time.time()
            p = run_model(m, lm, crops, starts, device, sig)
            p[~valid] = np.nan
            results[key] = p
            print(f"  {key} model: {len(starts)} windows ({time.time() - t1:.0f} s)")
        del crops
        remove_tmp(tmp.name)

        # ---- per-window CSV
        with open(out_dir / f"{vid.stem}_timeline.csv", "w", newline="") as f:
            w = csv.writer(f)
            head = ["start_s", "end_s", "centre_s", "face_share"]
            for key, _, names, *_ in models:
                head += [f"{key}_{c}" for c in names]
            w.writerow(head)
            for i, s in enumerate(starts):
                row = [f"{times[s]:.2f}", f"{times[s + 15]:.2f}", f"{centres[i]:.2f}", f"{face_share[i]:.2f}"]
                for key, *_ in models:
                    row += ["" if not np.isfinite(v) else f"{v:.4f}" for v in results[key][i]]
                w.writerow(row)

        # ---- segments + figure
        seg_rows = []
        fig, axes = plt.subplots(len(models), 1, figsize=(14, 3.6 * len(models)), sharex=True, squeeze=False)
        for ax, (key, title, names, sig, _, _, min_score) in zip(axes[:, 0], models):
            sm = smooth(results[key], args.smooth)
            labels = dominant(sm, names, min_score)
            segs = segments(labels, centres, args.stride, args.min_segment)
            for lab, a, b in segs:
                seg_rows.append([key, lab, f"{a:.1f}", f"{b:.1f}", fmt_time(a), fmt_time(b), f"{b - a:.1f}"])

            shown = ER_CLASSES if key == "er" else SHOWN_MAFW
            # no-face shading
            for i in np.where(~valid)[0]:
                ax.axvspan(centres[i] - args.stride / 2, centres[i] + args.stride / 2, color=NO_FACE, lw=0, zorder=0)
            for emo in shown:
                j = names.index(emo)
                ax.plot(centres, sm[:, j], color=STYLE[emo][0], lw=1.6, label=emo, zorder=2)
            if key == "mafw":
                rare = np.nansum(sm[:, [names.index(c) for c in RARE_MAFW]], axis=1)
                rare[~np.isfinite(sm).all(1)] = np.nan
                ax.plot(centres, rare, color=RARE_COLOR, lw=1.3, ls=(0, (4, 2)), zorder=2,
                        label="other 4 (anxiety, contempt,\ndisappointment, helplessness)")
            if key == "er":
                ax.axhline(min_score, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
            # dominant-emotion band above the plot
            for lab, a, b in segs:
                col = (STYLE[lab][0] if lab in STYLE else RARE_COLOR if lab in RARE_MAFW
                       else NO_FACE if lab == "no face" else NONE_COLOR)
                ax.axvspan(a, b, ymin=1.02, ymax=1.10, color=col, clip_on=False, lw=0)
                if b - a > duration / 25:
                    ax.text((a + b) / 2, 1.06, lab, transform=ax.get_xaxis_transform(), ha="center", va="center",
                            fontsize=7.5, color="white" if (lab in STYLE or lab in RARE_MAFW) else TEXT,
                            clip_on=False)
            ax.set_ylim(0, 1)
            ax.set_ylabel("score (smoothed)", fontsize=9, color=MUTED)
            ax.set_title(title, fontsize=10.5, color=TEXT, loc="left", pad=22)
            ax.grid(axis="y", color=GRID, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#d3d2cc")
            ax.tick_params(colors=MUTED, labelsize=8.5)
            ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False, fontsize=8.5, labelcolor=TEXT)
        axes[-1, 0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_time))
        axes[-1, 0].set_xlabel("time in video (min:s)  ·  shaded = no face detected", fontsize=9, color=MUTED)
        axes[-1, 0].set_xlim(0, duration)
        fig.suptitle(f"{vid.name}  ·  {args.window:g} s windows every {args.stride:g} s, "
                     f"smoothed over {args.smooth} windows", fontsize=12, color=TEXT, x=0.01, ha="left")
        fig.tight_layout()
        png = out_dir / f"{vid.stem}_timeline.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        with open(out_dir / f"{vid.stem}_segments.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "dominant_emotion", "start_s", "end_s", "start", "end", "duration_s"])
            w.writerows(seg_rows)
        print(f"  saved {png} ({time.time() - t0:.0f} s total)")


if __name__ == "__main__":
    main()
