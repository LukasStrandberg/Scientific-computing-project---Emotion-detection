"""Long patient videos: windows, face crops on a shared grid, smoothing, dominant label, segments.

Each window is treated exactly like an ElderReact test clip: 32 frames on a uniform grid with
step window / 32, face-cropped like the training data. Overlapping windows share their crops, so a
whole video is cropped in one pass; crops are kept in a temporary memory-mapped file on disk.
The stride is rounded to a whole number of grid steps (s2d preset: 3 s / 32 = 0.094 s steps,
11 steps = 1.03 s). The same windows are used for all three tracks.

Presets (config patient.presets): s2d = 3 s windows every 1 s, moving average over 5 windows
(default); maedfer = 1 s windows every 0.8 s, unsmoothed.
"""
import time
from collections import Counter

import cv2
import numpy as np

from common.faces import align_crop

NO_FACE, NO_CLEAR = "no face", "no clear emotion"


def window_settings(cfg, preset=None):
    p = cfg.patient.presets[preset or cfg.patient.preset]
    n = cfg.cache.n_frames
    step = p["window"] / n
    k = max(int(round(p["stride"] / step)), 1)
    return dict(window=p["window"], stride=k * step, smooth=int(p["smooth"]), step=step, k=k, n=n)


def grid_times(duration, step):
    return np.arange(0, duration - 1e-9, step)


def window_starts(n_grid, k, n=32):
    return list(range(0, n_grid - n + 1, k))


def expected_windows(duration, ws):
    """Number of windows for a video of `duration` seconds (used by the tests)."""
    return len(window_starts(len(grid_times(duration, ws["step"])), ws["k"], ws["n"]))


def crop_video_grid(cfg, video, step, detector, tmp_path, batch=64, log_every_min=2.0):
    """Face crops every `step` seconds. Returns dict(times, crops (memmap uint8 N,S,S,3), has_face,
    boxes (N,4), duration, fps). Frames before the first detected face stay black (no face);
    later frames without a face reuse the last detected box."""
    size = cfg.cache.size
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    wanted = sorted({int(round(t * fps)) for t in grid_times(duration, step)} - {n_frames})
    wanted_set = set(wanted)
    crops = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.uint8, shape=(len(wanted), size, size, 3))
    times, has_face, boxes, frames = [], [], [], []
    state = {"count": 0, "last": None}
    t0, next_log = time.time(), log_every_min

    def flush():
        for fr, d in zip(frames, detector(frames)):
            has_face.append(d is not None)
            if d is not None:
                state["last"] = d
            last = state["last"]
            if last is not None:
                style = cfg.faces.style
                crops[state["count"]] = cv2.cvtColor(align_crop(fr, last[0], last[1], size, cfg.faces.margin, style),
                                                     cv2.COLOR_BGR2RGB)
                boxes.append(last[0])
            else:
                boxes.append(np.full(4, np.nan, np.float32))
            state["count"] += 1
        frames.clear()

    i = 0
    while i < n_frames:
        if i in wanted_set:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
            times.append(i / fps)
            if len(frames) == batch:
                flush()
                if times[-1] / 60 >= next_log:
                    print(f"    {times[-1] / 60:.1f} / {duration / 60:.1f} min of video ({time.time() - t0:.0f} s)")
                    next_log += log_every_min
        elif not cap.grab():
            break
        i += 1
    if frames:
        flush()
    cap.release()
    count = state["count"]
    return dict(times=np.array(times), crops=crops[:count], has_face=np.array(has_face, bool),
                boxes=np.array(boxes, np.float32).reshape(-1, 4), duration=duration, fps=fps)


def window_table(grid, ws, min_face):
    starts = window_starts(len(grid["crops"]), ws["k"], ws["n"])
    t = grid["times"]
    start_s = np.array([t[s] for s in starts])
    end_s = np.array([t[s + ws["n"] - 1] + ws["step"] for s in starts])
    face_share = np.array([grid["has_face"][s:s + ws["n"]].mean() for s in starts])
    return dict(starts=starts, start_s=start_s, end_s=end_s, centre_s=(start_s + end_s) / 2,
                face_share=face_share, valid=face_share >= min_face)


def iter_windows(crops, starts, n=32, batch=8):
    for b in range(0, len(starts), batch):
        yield b, np.stack([np.asarray(crops[s:s + n]) for s in starts[b:b + batch]])


# ------------------------------------------------------------------ smoothing / labels
def smooth(x, k):
    """Centred moving average over k windows (ignores NaN = no-face windows)."""
    x = np.asarray(x, dtype=float)
    if k <= 1:
        return x.copy()
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - k // 2): i + k // 2 + 1]
        ok = np.isfinite(seg).all(axis=-1) if seg.ndim > 1 else np.isfinite(seg)
        if ok.any() and np.isfinite(x[i]).all():
            out[i] = seg[ok].mean(0)
    return out


def smooth_labels(labels, k):
    """Centred majority vote over k windows (for cluster ids); NO_FACE windows are kept."""
    if k <= 1:
        return list(labels)
    out = []
    for i, lab in enumerate(labels):
        if lab == NO_FACE:
            out.append(lab)
            continue
        seg = [v for v in labels[max(0, i - k // 2): i + k // 2 + 1] if v != NO_FACE]
        out.append(Counter(seg).most_common(1)[0][0])
    return out


def dominant(scores, names, min_score):
    lab = []
    for row in scores:
        if not np.isfinite(row).all():
            lab.append(NO_FACE)
        elif row.max() < min_score:
            lab.append(NO_CLEAR)
        else:
            lab.append(names[int(np.argmax(row))])
    return lab


def segments(labels, centres, stride, min_len):
    """Merge runs of equal labels; runs shorter than min_len seconds are absorbed by the longer
    neighbour. Returns [(label, start_s, end_s)]."""
    runs = []
    for lab, t in zip(labels, centres):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = t
        else:
            runs.append([lab, t, t])
    changed = True
    while changed and len(runs) > 1:
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
        k = 1
        while k < len(runs):
            if runs[k][0] == runs[k - 1][0]:
                runs[k - 1][2] = runs[k][2]
                del runs[k]
            else:
                k += 1
    return [(lab, max(a - stride / 2, 0.0), b + stride / 2) for lab, a, b in runs]


def summary(labels, segs, stride, duration):
    """Same patient summary for every track."""
    n = max(len(labels), 1)
    share = {lab: c / n for lab, c in Counter(labels).most_common()}
    changes = max(len(segs) - 1, 0)
    lengths = [b - a for _, a, b in segs]
    return {"windows": len(labels), "duration_s": float(duration), "stride_s": float(stride),
            "time_share": share, "segments": len(segs),
            "changes_per_minute": changes / max(duration / 60, 1e-9),
            "mean_segment_s": float(np.mean(lengths)) if lengths else 0.0}


def fmt_time(seconds, _pos=None):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"
