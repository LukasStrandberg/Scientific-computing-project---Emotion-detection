"""ElderReact labels and the one shared train / val / test split.

Label files (ElderReact repo, Annotations/): `<clip>.mp4 a d f h s su <gender> <valence>`, six
0/1 emotions in the order anger, disgust, fear, happiness, sadness, surprise.

Shared split (all tracks): official train + dev are pooled in label-file order, shuffled with
random.Random(seed).shuffle and the first round(val_frac * n) clips become validation. This is the
rule of the MAE-DFER track and reproduces its validation set exactly. The official test set is
never touched. The result is shipped as splits/elderreact_pooled_seed42.csv (`video,split`).
"""
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
OFFICIAL = ("train", "dev", "test")


def stem(name):
    return Path(str(name)).stem if Path(str(name)).suffix.lower() in VIDEO_EXTS else Path(str(name)).name


def read_labels(path):
    """[(clip_stem, [6 ints])] in file order."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 7:
                continue
            try:
                y = [int(float(v)) for v in parts[1:7]]
            except ValueError:   # header line
                continue
            rows.append((stem(parts[0]), y))
    return rows


def make_pooled_split(train_rows, dev_rows, test_rows, val_frac=0.15, seed=42):
    """dict clip -> 'train' | 'val' | 'test' (see module docstring)."""
    pool = [c for c, _ in train_rows] + [c for c, _ in dev_rows]
    random.Random(seed).shuffle(pool)
    n_val = int(round(val_frac * len(pool)))
    split = {c: "val" for c in pool[:n_val]}
    split.update({c: "train" for c in pool[n_val:]})
    split.update({c: "test" for c, _ in test_rows})
    return split


def label_files(cfg):
    root = cfg.path("elderreact")
    return {part: root / f"{part}_labels.txt" for part in OFFICIAL}


def write_split(path, split):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["video", "split"])
        w.writerows(split.items())


def index_videos(folder):
    """clip stem -> video path, searched recursively."""
    out = {}
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS:
            out.setdefault(p.stem, p)
    return out


def load_split(cfg, require_videos=True):
    """DataFrame: video, split, official, <6 emotions>, path. Creates the split file if missing and
    checks a shipped split file against the label files."""
    files = label_files(cfg)
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"ElderReact label files not found: {missing}. Put train/dev/test_labels.txt "
                                f"in {cfg.path('elderreact')} (see README, 'Data layout').")
    rows = {part: read_labels(p) for part, p in files.items()}
    split = make_pooled_split(rows["train"], rows["dev"], rows["test"], cfg.split.val_frac, cfg.split.seed)
    split_file = Path(cfg.paths.split_file)
    if split_file.exists():
        shipped = dict(pd.read_csv(split_file).astype(str).values)
        if shipped != split:
            raise RuntimeError(f"{split_file} does not match the split computed from the label files "
                               f"(different label files or split settings). Delete it to regenerate.")
    else:
        write_split(split_file, split)
        print(f"wrote split file {split_file}")
    emotions = list(cfg.emotions)
    recs = [dict(video=c, split=split[c], official=part, **dict(zip(emotions, y)))
            for part in OFFICIAL for c, y in rows[part]]
    df = pd.DataFrame(recs)
    videos = index_videos(cfg.path("elderreact"))
    df["path"] = [str(videos.get(c, "")) for c in df.video]
    n_missing = int((df.path == "").sum())
    if n_missing:
        msg = f"{n_missing} labelled clips have no video file under {cfg.path('elderreact')}"
        if require_videos:
            raise FileNotFoundError(msg)
        print("Note:", msg)
    return df


def labels_of(df, emotions):
    return df[list(emotions)].to_numpy(dtype=np.float32)


def split_summary(df, emotions):
    out = []
    for part in ("train", "val", "test"):
        d = df[df.split == part]
        y = labels_of(d, emotions)
        out.append(dict(split=part, clips=len(d), mean_labels=float(y.sum(1).mean()) if len(d) else 0.0,
                        no_label=int((y.sum(1) == 0).sum()), single=int((y.sum(1) == 1).sum()),
                        multi=int((y.sum(1) >= 2).sum()), **{e: int(y[:, i].sum()) for i, e in enumerate(emotions)}))
    return pd.DataFrame(out).set_index("split")
