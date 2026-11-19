"""Evaluate an S2D checkpoint on face-cropped clips and save predictions + features.

Default = official S2D test protocol: all frames of a clip are used, two windows of 16
consecutive frames are taken (start and end of the clip) and the class probabilities of
the windows are averaged. Runs on one GPU on Windows, needs no distributed setup, and
also saves the 768-d clip features for t-SNE.

Modes (--mode):
    mafw         11-class MAFW model on MAFW (single label): UAR, WAR, per-class recall
    er-zeroshot  11-class MAFW model on ElderReact: the model's probabilities for the 6
                 emotions ElderReact shares with MAFW are used as scores (no training)
    er           6-output model fine-tuned on ElderReact (sigmoid outputs)
    ElderReact metrics: AUC and F1 per emotion (+ means), and accuracy on the clips that
    have exactly one emotion. F1 needs a yes/no threshold per emotion: pass thresholds
    tuned on the dev set with --thresholds; without it they are tuned on the evaluated
    set itself (fine for dev, optimistic for test) and written to thresholds.json.

Frame sampling (--sampling):
    consecutive  official: windows of 16 consecutive frames, evenly spaced from start to end
    uniform      each window takes 16 frames spread evenly over the WHOLE clip
                 (windows are shifted slightly against each other)

Examples (from the repo root with venv-s2d active):
    python s2d/evaluate.py --ckpt external/S2D/ckpts/MAFW/without_oversampler/split01_checkpoint-best.pth ^
        --anno data/processed/MAFW/anno/set_1/test.csv --frames data/processed/MAFW/test_deepface ^
        --out results/mafw_fold1_uniform --sampling uniform

    python s2d/evaluate.py --mode er-zeroshot --ckpt <same> ^
        --anno data/processed/ElderReact/anno/dev.csv --frames data/processed/ElderReact/faces_224 ^
        --out results/er_zeroshot_dev

Outputs in --out:
    predictions.csv   clip, (true label), predicted label / yes-no, score per class
    features.npz      features (N x 768), labels, scores, clip ids  -> t-SNE
    metrics.txt       metrics as described above
    thresholds.json   (ElderReact modes) thresholds used for F1
"""
import argparse
import csv
import json
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
ER_CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
ER_IN_MAFW = [MAFW_CLASSES.index(c) for c in ER_CLASSES]  # [0, 4, 5, 6, 9, 10]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- frame sampling
def window_indices(n_frames, clip_len, n_windows, sampling):
    """List of n_windows index lists (each clip_len long) into a clip of n_frames frames."""
    if sampling == "consecutive":  # official S2D test
        step = max((n_frames - clip_len) / (n_windows - 1), 0) if n_windows > 1 else 0
        starts = [int(i * step) for i in range(n_windows)]
        return [[min(s + t, n_frames - 1) for t in range(clip_len)] for s in starts]
    # uniform: split the clip into clip_len equal bins, take one frame per bin;
    # window k takes the frame at relative position (k + 0.5) / n_windows inside each bin
    bin_len = n_frames / clip_len
    return [[min(int((t + (k + 0.5) / n_windows) * bin_len), n_frames - 1) for t in range(clip_len)]
            for k in range(n_windows)]


def load_clip_windows(frame_paths, clip_len=16, n_windows=2, sampling="consecutive", size=224):
    """Returns tensor (n_windows, 3, clip_len, size, size), normalised like S2D."""
    idx = window_indices(len(frame_paths), clip_len, n_windows, sampling)
    cache = {}
    for i in sorted({i for w in idx for i in w}):
        img = Image.open(frame_paths[i]).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        cache[i] = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    x = np.stack([np.stack([cache[i] for i in w]) for w in idx])  # W T H W C
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3)))  # W C T H W


class ClipDataset(Dataset):
    def __init__(self, anno_csv, frames_root, limit=0, clip_len=16, n_windows=2, sampling="consecutive"):
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
        self.clip_len, self.n_windows, self.sampling = clip_len, n_windows, sampling

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        row, d = self.items[i]
        frames = sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))
        x = load_clip_windows(frames, self.clip_len, self.n_windows, self.sampling)
        name, labels = row
        return x, torch.tensor(labels, dtype=torch.float32), name


# ---------------------------------------------------------------- model
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


# ---------------------------------------------------------------- metrics
def recall_metrics(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    support = cm.sum(1)
    recall = np.divide(np.diag(cm), support, out=np.full(n_classes, np.nan), where=support > 0)
    return np.nanmean(recall), np.trace(cm) / max(cm.sum(), 1), recall, cm


def auc(y, s):
    """ROC AUC via the rank (Mann-Whitney) formula; nan if only one class present."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = np.asarray(s)[order]
    i = 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, s):
    """Average precision (area under the precision-recall curve), same definition as
    sklearn.metrics.average_precision_score; nan if the emotion never occurs."""
    y, s = np.asarray(y).astype(bool), np.asarray(s, dtype=float)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]  # last index of each group of equal scores
    tp = np.cumsum(y)[last]
    precision = tp / (last + 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def f1(y, yhat):
    y, yhat = np.asarray(y).astype(bool), np.asarray(yhat).astype(bool)
    tp = (y & yhat).sum()
    return 2 * tp / max(y.sum() + yhat.sum(), 1)


def best_threshold(y, s):
    cands = np.unique(np.concatenate([s, [0.5]]))
    scores = [f1(y, s >= t) for t in cands]
    return float(cands[int(np.argmax(scores))])


def multilabel_report(labels, scores, names, thresholds=None):
    """labels, scores: (N, C). Returns (text lines, thresholds used, yes/no predictions)."""
    tuned_here = thresholds is None
    if tuned_here:
        thresholds = {c: best_threshold(labels[:, i], scores[:, i]) for i, c in enumerate(names)}
    yes = np.stack([scores[:, i] >= thresholds[c] for i, c in enumerate(names)], 1)
    lines = ["Per emotion (n = clips where the emotion is present):",
             f"  {'emotion':12s} {'n':>5s} {'AUC':>7s} {'AP':>7s} {'F1':>7s} {'F1@0.5':>7s} {'threshold':>10s}"]
    aucs, aps, f1s, f05s = [], [], [], []
    for i, c in enumerate(names):
        a, p = auc(labels[:, i], scores[:, i]), average_precision(labels[:, i], scores[:, i])
        f, f05 = f1(labels[:, i], yes[:, i]), f1(labels[:, i], scores[:, i] >= 0.5)
        aucs.append(a); aps.append(p); f1s.append(f); f05s.append(f05)
        lines.append(f"  {c:12s} {int(labels[:, i].sum()):5d} {100 * a:7.2f} {100 * p:7.2f} {100 * f:7.2f}"
                     f" {100 * f05:7.2f} {thresholds[c]:10.4f}")
    lines += [f"  {'MEAN':12s} {'':5s} {100 * np.nanmean(aucs):7.2f} {100 * np.nanmean(aps):7.2f}"
              f" {100 * np.mean(f1s):7.2f} {100 * np.mean(f05s):7.2f}",
              "  AUC = ranking quality (50 = chance); AP = area under precision-recall (mean = mAP);",
              "  F1 uses the thresholds in the last column; F1@0.5 uses a fixed 0.5 cutoff (MAE-DFER track).",
              ("  (thresholds tuned on THIS set - fine for validation, optimistic for test)" if tuned_here
               else "  (thresholds loaded from --thresholds)")]
    summary = {"mean_auc": float(np.nanmean(aucs)), "mAP": float(np.nanmean(aps)),
               "mean_f1": float(np.mean(f1s)), "mean_f1_at_0.5": float(np.mean(f05s)),
               "auc": dict(zip(names, map(float, aucs))), "ap": dict(zip(names, map(float, aps))),
               "f1": dict(zip(names, map(float, f1s))), "f1_at_0.5": dict(zip(names, map(float, f05s)))}
    one = labels.sum(1) == 1
    if one.any():
        y1, p1 = labels[one].argmax(1), scores[one].argmax(1)
        uar, war, recall, cm = recall_metrics(y1, p1, len(names))
        lines += ["", f"Clips with exactly one emotion: {int(one.sum())} (side metric, few clips per class)",
                  f"  accuracy (WAR): {100 * war:.2f}   mean per-emotion recall (UAR): {100 * uar:.2f}"]
        lines += [f"    {c:12s} {100 * r:6.2f}  (n={n})" if not np.isnan(r) else f"    {c:12s}    -    (n=0)"
                  for c, r, n in zip(names, recall, cm.sum(1))]
        summary.update(single_war=float(war), single_uar=float(uar))
    return lines, thresholds, yes, summary


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anno", required=True, help="CSV from make_annotations.py")
    ap.add_argument("--frames", required=True, help="folder with one face-crop folder per clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="mafw", choices=["mafw", "er-zeroshot", "er"])
    ap.add_argument("--sampling", default="consecutive", choices=["consecutive", "uniform"])
    ap.add_argument("--windows", type=int, default=2, help="16-frame windows per clip (official: 2)")
    ap.add_argument("--thresholds", help="thresholds.json from a dev-set run (ElderReact modes)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
    ap.add_argument("--batch", type=int, default=4, help="clips per batch")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU (slow)")
    ds = ClipDataset(args.anno, args.frames, args.limit, n_windows=args.windows, sampling=args.sampling)
    if len(ds) == 0:
        sys.exit("No clips with face crops found - run preprocess_faces.py first.")
    num_classes = 6 if args.mode == "er" else 11
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=device == "cuda")
    model, landmark = build_model(args.ckpt, num_classes, device)

    ids, labels, probs, feats = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for b, (x, y, names) in enumerate(dl, 1):
            B, W = x.shape[:2]
            x = x.flatten(0, 1).to(device, non_blocking=True)  # (B*W, C, T, H, W)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits, feat = model(x, landmark(x))
            logits = logits.float()
            p = logits.sigmoid() if args.mode == "er" else logits.softmax(-1)
            probs.append(p.view(B, W, -1).mean(1).cpu().numpy())
            feats.append(feat.float().view(B, W, -1).mean(1).cpu().numpy())
            labels.append(y.numpy()); ids += list(names)
            if b % 10 == 0 or b == len(dl):
                print(f"  batch {b}/{len(dl)}  ({time.time() - t0:.0f} s)")
    probs, feats, labels = np.concatenate(probs), np.concatenate(feats), np.concatenate(labels)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    header = [f"checkpoint: {args.ckpt}", f"annotations: {args.anno}", f"frames: {args.frames}",
              f"mode: {args.mode}   sampling: {args.sampling}   windows: {args.windows}",
              f"clips evaluated: {len(ids)}", f"time: {time.time() - t0:.0f} s", ""]

    if args.mode == "mafw":
        names, pred = MAFW_CLASSES, probs.argmax(1)
        y_true = labels[:, 0].astype(int)
        uar, war, recall, cm = recall_metrics(y_true, pred, len(names))
        lines = [f"UAR: {100 * uar:.2f}", f"WAR: {100 * war:.2f}", "", "Recall per class:"]
        lines += [f"  {c:15s} {100 * r:6.2f}  (n={n})" if not np.isnan(r) else f"  {c:15s}    -    (n=0)"
                  for c, r, n in zip(names, recall, cm.sum(1))]
        lines += ["", "Confusion matrix (rows = true, cols = predicted, order as above):",
                  np.array2string(cm, max_line_width=200)]
        with open(out / "predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip", "true", "pred"] + [f"p_{c}" for c in names])
            for i, clip in enumerate(ids):
                w.writerow([clip, names[y_true[i]], names[pred[i]]] + [f"{v:.4f}" for v in probs[i]])
        np.savez(out / "features.npz", features=feats, labels=labels, pred=pred, probs=probs,
                 clips=np.array(ids), class_names=np.array(names))
    else:
        names = ER_CLASSES
        scores = probs[:, ER_IN_MAFW] if args.mode == "er-zeroshot" else probs
        thr = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
        lines, thr, yes, summary = multilabel_report(labels, scores, names, thr)
        if args.mode == "er-zeroshot":
            lines.insert(3 + len(names), "  (zero-shot: scores are softmax shares of 11 MAFW classes, so F1@0.5 is "
                                         "not meaningful - use AUC / AP / tuned F1)")
        (out / "thresholds.json").write_text(json.dumps(thr, indent=2))
        (out / "metrics.json").write_text(json.dumps(summary, indent=2))
        with open(out / "predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip"] + [f"true_{c}" for c in names] + [f"pred_{c}" for c in names]
                       + [f"score_{c}" for c in names])
            for i, clip in enumerate(ids):
                w.writerow([clip] + [int(v) for v in labels[i]] + [int(v) for v in yes[i]]
                           + [f"{v:.4f}" for v in scores[i]])
        np.savez(out / "features.npz", features=feats, labels=labels, scores=scores, probs=probs,
                 clips=np.array(ids), class_names=np.array(names))

    text = "\n".join(header + lines)
    (out / "metrics.txt").write_text(text)
    print("\n" + text + f"\n\nSaved to {out}")


if __name__ == "__main__":
    main()
