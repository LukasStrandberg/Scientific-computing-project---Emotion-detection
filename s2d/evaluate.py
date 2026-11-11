"""Evaluate an S2D checkpoint on face-cropped clips and save predictions + features.

Follows the official S2D test protocol: all frames of a clip are used, two windows of
16 consecutive frames are taken (start and end of the clip), and the class probabilities
of the two windows are averaged. Unlike the official script it runs on one GPU on
Windows, needs no distributed setup, and also saves the 768-d clip features for t-SNE.

Example (smoke test, from the repo root with venv-s2d active):
    python s2d/evaluate.py ^
        --ckpt external/S2D/ckpts/MAFW/without_oversampler/split01_checkpoint-best.pth ^
        --anno data/processed/MAFW/anno/set_1/test.csv ^
        --frames data/processed/MAFW/faces_224 ^
        --out results/smoke_mafw_fold1 --limit 50

Outputs in --out:
    predictions.csv   clip, true label, predicted label, probability per class
    features.npz      features (N x 768), labels, predictions, clip ids  -> t-SNE
    metrics.txt       UAR, WAR, per-class recall, confusion matrix
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
S2D_DIR = REPO / "external" / "S2D"
sys.path.insert(0, str(S2D_DIR))

MAFW_CLASSES = ["anger", "anxiety", "contempt", "disappointment", "disgust", "fear",
                "happiness", "helplessness", "neutral", "sadness", "surprise"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def window_starts(n_frames, clip_len, n_windows):
    """Same as official S2D test: evenly spaced start positions from 0 to n_frames-clip_len."""
    if n_windows == 1:
        return [0]
    step = max((n_frames - clip_len) / (n_windows - 1), 0)
    return [int(i * step) for i in range(n_windows)]


def load_clip_windows(frame_paths, clip_len=16, n_windows=2, size=224):
    """Returns tensor (n_windows, 3, clip_len, size, size), normalised like S2D."""
    frames = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        frames.append(np.asarray(img, dtype=np.float32) / 255.0)
    while len(frames) < clip_len:  # very short clip: repeat the last frame
        frames.append(frames[-1])
    arr = (np.stack(frames) - MEAN) / STD  # T H W C
    wins = [arr[s:s + clip_len] for s in window_starts(len(arr), clip_len, n_windows)]
    x = np.stack(wins).transpose(0, 4, 1, 2, 3)  # W C T H W
    return torch.from_numpy(np.ascontiguousarray(x))


class ClipDataset(Dataset):
    def __init__(self, anno_csv, frames_root, limit=0, clip_len=16, n_windows=2):
        with open(anno_csv, newline="") as f:
            rows = list(csv.reader(f))
        self.label_cols = rows[0][1:]
        root = Path(frames_root)
        have = [((r[0], [float(v) for v in r[1:]]), root / r[0]) for r in rows[1:] if r]
        missing = [r[0] for r, d in have if not d.is_dir()]
        have = [(r, d) for r, d in have if d.is_dir()]
        if missing:
            print(f"Note: {len(missing)} clips in {anno_csv} have no face folder yet and are skipped.")
        if limit:
            have = have[:limit]
        self.items = have
        self.clip_len, self.n_windows = clip_len, n_windows

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        row, d = self.items[i]
        frames = sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))
        x = load_clip_windows(frames, self.clip_len, self.n_windows)
        name, labels = row
        return x, torch.tensor(labels, dtype=torch.float32), name


def build_model(ckpt_path, num_classes, device):
    import model_finetuning  # noqa: F401  (registers the S2D model with timm)
    from timm.models import create_model
    from landmark.backbone import Landmark

    model = create_model("s2d_base_patch16_224", pretrained=False, num_classes=num_classes,
                         adapter_scale=0.25, head_dropout_ratio=0.5, num_frames=16, in_chans_l=128)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {ckpt_path}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        print("  missing (first 5):", missing[:5])
    landmark = Landmark("MobileFaceNet")
    return model.to(device).eval(), landmark.to(device).eval()


def recall_metrics(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    support = cm.sum(1)
    recall = np.divide(np.diag(cm), support, out=np.full(n_classes, np.nan), where=support > 0)
    uar = np.nanmean(recall)
    war = np.trace(cm) / cm.sum()
    return uar, war, recall, cm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anno", required=True, help="CSV from make_annotations.py")
    ap.add_argument("--frames", required=True, help="folder with one face-crop folder per clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
    ap.add_argument("--windows", type=int, default=2, help="16-frame windows per clip (official: 2)")
    ap.add_argument("--batch", type=int, default=4, help="clips per batch")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--num-classes", type=int, default=11)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU (slow)")
    ds = ClipDataset(args.anno, args.frames, args.limit, n_windows=args.windows)
    if len(ds) == 0:
        sys.exit("No clips with face crops found - run preprocess_faces.py first.")
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=device == "cuda")
    model, landmark = build_model(args.ckpt, args.num_classes, device)

    ids, labels, probs, feats = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for b, (x, y, names) in enumerate(dl, 1):
            B, W = x.shape[:2]
            x = x.flatten(0, 1).to(device, non_blocking=True)  # (B*W, C, T, H, W)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits, feat = model(x, landmark(x))
            p = logits.float().softmax(-1).view(B, W, -1).mean(1)
            f = feat.float().view(B, W, -1).mean(1)
            probs.append(p.cpu().numpy()); feats.append(f.cpu().numpy())
            labels.append(y.numpy()); ids += list(names)
            if b % 10 == 0 or b == len(dl):
                print(f"  batch {b}/{len(dl)}  ({time.time() - t0:.0f} s)")
    probs, feats, labels = np.concatenate(probs), np.concatenate(feats), np.concatenate(labels)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    class_names = MAFW_CLASSES if args.num_classes == 11 else [f"c{i}" for i in range(args.num_classes)]
    pred = probs.argmax(1)
    single = labels.shape[1] == 1  # single-label dataset (MAFW)
    if single:
        y_true = labels[:, 0].astype(int)
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip"] + (["true"] if single else []) + ["pred"] + [f"p_{c}" for c in class_names])
        for i, clip in enumerate(ids):
            w.writerow([clip] + ([class_names[y_true[i]]] if single else []) + [class_names[pred[i]]]
                       + [f"{v:.4f}" for v in probs[i]])
    np.savez(out / "features.npz", features=feats, labels=labels, pred=pred, probs=probs,
             clips=np.array(ids), class_names=np.array(class_names))

    lines = [f"checkpoint: {args.ckpt}", f"annotations: {args.anno}", f"clips evaluated: {len(ids)}",
             f"time: {time.time() - t0:.0f} s"]
    if labels.shape[1] == 1:
        uar, war, recall, cm = recall_metrics(y_true, pred, args.num_classes)
        lines += [f"UAR: {100 * uar:.2f}", f"WAR: {100 * war:.2f}", "", "Recall per class:"]
        lines += [f"  {c:15s} {100 * r:6.2f}  (n={n})" if not np.isnan(r) else f"  {c:15s}    -    (n=0)"
                  for c, r, n in zip(class_names, recall, cm.sum(1))]
        lines += ["", "Confusion matrix (rows = true, cols = predicted, order as above):",
                  np.array2string(cm, max_line_width=200)]
    text = "\n".join(lines)
    (out / "metrics.txt").write_text(text)
    print("\n" + text + f"\n\nSaved to {out}")


if __name__ == "__main__":
    main()
