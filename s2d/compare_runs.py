"""Compare ElderReact test results of several models, with 95 % bootstrap confidence intervals.

Input per model: an evaluate.py output folder (ElderReact modes er-zeroshot / er), or any
CSV in the shared layout below - so the MAE-DFER model can be compared the same way.

Shared predictions CSV layout (one row per test clip):
    clip, true_anger, ..., true_surprise, score_anger, ..., score_surprise [, pred_anger, ..., pred_surprise]
    * clip: video name without extension (e.g. 50_50_17)
    * true_*: 0/1 labels from test_labels.txt
    * score_*: the model's probability (sigmoid) per emotion
    * pred_* (optional): 0/1 decisions with thresholds tuned on the validation set.
      Without pred_* columns the "F1 (tuned)" column is computed with a 0.5 cutoff.
    Emotion order: anger, disgust, fear, happiness, sadness, surprise.

Metrics per model (means over the 6 emotions):
    AUC        ranking quality, 50 = chance
    mAP        mean average precision (area under precision-recall), chance = base rate
    F1 (tuned) F1 with the validation-tuned thresholds (pred_* columns)
    F1@0.5     F1 with a fixed 0.5 cutoff (the MAE-DFER track's main F1)

Confidence intervals: the test clips are resampled with replacement (default 2000 times),
all metrics are recomputed on every resample, and the 2.5th / 97.5th percentiles are
reported. The same resamples are used for all models, so the differences are paired: if a
difference's interval excludes 0 (marked *), it is unlikely to be chance on this test set.

Example:
    python s2d/compare_runs.py ^
        --runs results/er_zeroshot_uni_test results/er_ft_pooled_test results/maedfer_test.csv ^
        --names "S2D before" "S2D fine-tuned" "MAE-DFER" --out results/compare_elderreact.txt
"""
import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")  # emotions absent from a resample
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import ER_CLASSES, auc, average_precision, f1  # noqa: E402

METRICS = [("AUC", "auc"), ("mAP", "ap"), ("F1 (tuned)", "f1"), ("F1@0.5", "f05")]


def load(path):
    p = Path(path)
    if p.is_dir():
        p = p / "predictions.csv"
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    clips = [Path(r["clip"]).stem for r in rows]
    y = np.array([[int(float(r[f"true_{c}"])) for c in ER_CLASSES] for r in rows])
    s = np.array([[float(r[f"score_{c}"]) for c in ER_CLASSES] for r in rows])
    if all(f"pred_{c}" in rows[0] for c in ER_CLASSES):
        yes = np.array([[int(float(r[f"pred_{c}"])) for c in ER_CLASSES] for r in rows])
    else:
        print(f"{p}: no pred_* columns - 'F1 (tuned)' uses a 0.5 cutoff for this model")
        yes = (s >= 0.5).astype(int)
    return clips, y, yes, s


def metrics(y, yes, s):
    """Per-emotion arrays for every metric."""
    return {
        "auc": np.array([auc(y[:, i], s[:, i]) for i in range(y.shape[1])]),
        "ap": np.array([average_precision(y[:, i], s[:, i]) for i in range(y.shape[1])]),
        "f1": np.array([f1(y[:, i], yes[:, i]) for i in range(y.shape[1])]),
        "f05": np.array([f1(y[:, i], s[:, i] >= 0.5) for i in range(y.shape[1])]),
    }


def ci(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="evaluate.py output folders or prediction CSVs; the first one is the reference")
    ap.add_argument("--names", nargs="+", help="short name per run")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="also write the report to this text file")
    args = ap.parse_args()
    names = args.names or [Path(r).stem for r in args.runs]

    data = [load(r) for r in args.runs]
    ref_clips = data[0][0]
    aligned = []
    for (clips, y, yes, s), name in zip(data, names):
        if clips != ref_clips:  # align to the reference order
            pos = {c: i for i, c in enumerate(clips)}
            if set(pos) != set(ref_clips):
                only_ref = sorted(set(ref_clips) - set(pos))[:3]
                sys.exit(f"{name}: not the same clips as {names[0]} (e.g. missing {only_ref}) - "
                         f"evaluate all models on the same test set.")
            idx = [pos[c] for c in ref_clips]
            y, yes, s = y[idx], yes[idx], s[idx]
        aligned.append((y, yes, s))
    for (y, _, _), name in zip(aligned, names):
        if not np.array_equal(y, aligned[0][0]):
            sys.exit(f"{name}: labels differ from {names[0]}")

    n = len(ref_clips)
    rng = np.random.default_rng(args.seed)
    point = [metrics(*a) for a in aligned]
    boot = [{k: [] for _, k in METRICS} for _ in aligned]  # per model, per metric: list of per-emotion arrays
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        for k, (y, yes, s) in enumerate(aligned):
            m = metrics(y[idx], yes[idx], s[idx])
            for _, key in METRICS:
                boot[k][key].append(m[key])
    boot = [{key: np.array(v) for key, v in b.items()} for b in boot]  # (n_boot, 6)

    def mean_boot(k, key):
        return np.nanmean(boot[k][key], axis=1)

    lines = [f"ElderReact test: {n} clips, {args.n_boot} bootstrap resamples, 95 % intervals in brackets",
             f"(chance: AUC 50; mAP = mean base rate {100 * aligned[0][0].mean():.1f}. F1@0.5 is not meaningful"
             f" for the zero-shot MAFW model, whose 6 scores are shares of an 11-class softmax)", ""]
    lines.append(f"{'model':24s}" + "".join(f"{label:>23s}" for label, _ in METRICS))
    for k, name in enumerate(names):
        row = f"{name[:24]:24s}"
        for _, key in METRICS:
            lo, hi = ci(mean_boot(k, key))
            row += f"  {100 * np.nanmean(point[k][key]):5.1f} [{100 * lo:5.1f},{100 * hi:5.1f}]"
        lines.append(row)

    if len(names) > 1:
        lines += ["", f"Difference to '{names[0]}' (positive = better; * = interval excludes 0):",
                  f"{'model':24s}" + "".join(f"{label:>23s}" for label, _ in METRICS)]
        for k in range(1, len(names)):
            row = f"{names[k][:24]:24s}"
            for _, key in METRICS:
                d = mean_boot(k, key) - mean_boot(0, key)
                lo, hi = ci(d)
                star = "*" if not (lo <= 0 <= hi) else " "
                row += f" {100 * (np.nanmean(point[k][key]) - np.nanmean(point[0][key])):+5.1f} [{100 * lo:+5.1f},{100 * hi:+5.1f}]{star}"
            lines.append(row)

        for label, key in [("AUC", "auc"), ("AP", "ap")]:
            lines += ["", f"{label} per emotion (value; difference to '{names[0]}' with 95 % interval):"]
            lines.append(f"{'emotion':12s} {names[0][:16]:>16s}" + "".join(f" {nm[:28]:>28s}" for nm in names[1:]))
            for i, emo in enumerate(ER_CLASSES):
                row = f"{emo:12s} {100 * point[0][key][i]:16.1f}"
                for k in range(1, len(names)):
                    lo, hi = ci(boot[k][key][:, i] - boot[0][key][:, i])
                    d = point[k][key][i] - point[0][key][i]
                    row += f" {100 * point[k][key][i]:6.1f} ({100 * d:+5.1f} [{100 * lo:+5.1f},{100 * hi:+5.1f}])"
                lines.append(row)

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print("\nSaved", args.out)


if __name__ == "__main__":
    main()
