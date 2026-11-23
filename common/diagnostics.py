"""Face-detection diagnostic to confirm the detector / crop choice on real data.

For a random sample of clips it reports, per detector: share of the 32 sampled frames with a face,
clips without any face, and box jitter (std of the box centre over the clip / box size). It also
checks how often MediaPipe finds a face on the crop styles and on the full frame (this decides
`clustering.source`), and draws a contact sheet: middle frame per variant for a few clips.
"""
import numpy as np
import pandas as pd

from common import sampling
from common.faces import align_crop, make_detector


def run(cfg, df, n_clips=20, n_sheet=6, seed=0, device="cpu"):
    from clustering.blendshapes import make_extractor
    sample = df.sample(n=min(n_clips, len(df)), random_state=seed)
    detectors = {"retinaface": make_detector(cfg, device, "retinaface")}
    if cfg.resolve(cfg.faces.yolo_weights).exists() and not cfg.smoke_test:
        detectors["yolo"] = make_detector(cfg, device, "yolo")
    try:
        mp_ext = make_extractor(cfg)
    except FileNotFoundError as e:
        print("MediaPipe check skipped:", e)
        mp_ext = None
    size, rows, sheet = cfg.cache.size, [], []
    for clip, path in zip(sample.video, sample.path):
        frames, _ = sampling.read_all_frames(path)
        idx = sampling.uniform_indices(len(frames), cfg.cache.n_frames)
        fr = [frames[i] for i in idx]
        mid = len(fr) // 2
        row, crops = {"clip": clip}, {}
        for name, det in detectors.items():
            dets = det(fr)
            ok = [d for d in dets if d is not None]
            row[f"{name}_face_frames"] = len(ok) / len(fr)
            if ok:
                c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b, _ in ok])
                s = np.mean([max(b[2] - b[0], b[3] - b[1]) for b, _ in ok])
                row[f"{name}_jitter"] = float(c.std(0).mean() / s)
                b, k = dets[mid] if dets[mid] is not None else ok[0]
                for style in ("deepface", "square"):
                    crops[f"{name} {style}"] = align_crop(fr[mid], b, k, size, cfg.faces.margin, style)[..., ::-1]
        if mp_ext is not None:
            pick = fr[::4]
            row["mediapipe_full_frame"] = float(np.isfinite(mp_ext(np.stack([f[..., ::-1] for f in pick]))).all(1).mean())
            for name in list(crops):
                if name.startswith("retinaface"):
                    key = "mediapipe_" + name.split()[1]
                    sub = []
                    for f in pick:
                        d = detectors["retinaface"]([f])[0]
                        if d is not None:
                            sub.append(align_crop(f, d[0], d[1], size, cfg.faces.margin, name.split()[1])[..., ::-1])
                    row[key] = float(np.isfinite(mp_ext(np.stack(sub))).all(1).mean()) if sub else 0.0
        rows.append(row)
        if len(sheet) < n_sheet:
            sheet.append([fr[mid][..., ::-1]] + [crops.get(k) for k in sorted(crops)])
            sheet_cols = ["frame"] + sorted(crops)
    table = pd.DataFrame(rows)
    summary = {}
    for name in detectors:
        summary[f"{name}: frames with a face"] = table[f"{name}_face_frames"].mean()
        summary[f"{name}: clips without any face"] = int((table[f"{name}_face_frames"] == 0).sum())
        if f"{name}_jitter" in table:
            summary[f"{name}: box jitter (median)"] = table[f"{name}_jitter"].median()
    for col in [c for c in table.columns if c.startswith("mediapipe_")]:
        summary[f"MediaPipe face found on {col[10:].replace('_', ' ')}"] = table[col].mean()
    return table, pd.Series(summary, name="value").to_frame(), (sheet, sheet_cols if sheet else [])
