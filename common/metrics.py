"""Metrics for every track (multi-label ElderReact, 6 emotions) and for the clustering track.

Per emotion and as the mean over emotions:
    auc    ROC-AUC (50 % = chance)
    ap     average precision, same definition as sklearn.metrics.average_precision_score (mean = mAP)
    f1     F1 with a per-emotion threshold tuned on the VALIDATION split (never on test)
    f1_05  F1 with a fixed 0.5 cutoff
95 % confidence intervals: the clips are resampled with replacement (n_boot times, fixed seed) and
all metrics are recomputed; percentiles 2.5 / 97.5. Means ignore emotions that are absent from a
resample (NaN). In comparisons the same resamples are used for every run (paired bootstrap).
"""
import numpy as np
from scipy.stats import rankdata

METRICS = [("auc", "ROC-AUC"), ("ap", "AP"), ("f1", "F1 (val. thresholds)"), ("f1_05", "F1@0.5")]


# ------------------------------------------------------------------ single metrics
def auc(y, s):
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, s):
    y, s = np.asarray(y).astype(bool), np.asarray(s, dtype=float)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]   # last index of each group of tied scores
    tp = np.cumsum(y)[last]
    precision, recall = tp / (last + 1), tp / n_pos
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def f1(y, yhat):
    y, yhat = np.asarray(y).astype(bool), np.asarray(yhat).astype(bool)
    return float(2 * (y & yhat).sum() / max(y.sum() + yhat.sum(), 1))


def best_threshold(y, s):
    """Threshold (among the observed scores and 0.5) that maximises F1."""
    s = np.asarray(s, dtype=float)
    finite = s[np.isfinite(s)]
    cands = np.unique(np.concatenate([finite, [0.5]]))
    scores = [f1(y, s >= t) for t in cands]
    return float(cands[int(np.argmax(scores))])


def tune_thresholds(Y, S, names):
    return {n: best_threshold(Y[:, i], S[:, i]) for i, n in enumerate(names)}


def per_class(Y, S, thr, names):
    """dict metric -> array (C,)."""
    t = np.array([thr[n] for n in names])
    C = Y.shape[1]
    return {
        "auc": np.array([auc(Y[:, i], S[:, i]) for i in range(C)]),
        "ap": np.array([average_precision(Y[:, i], S[:, i]) for i in range(C)]),
        "f1": np.array([f1(Y[:, i], S[:, i] >= t[i]) for i in range(C)]),
        "f1_05": np.array([f1(Y[:, i], S[:, i] >= 0.5) for i in range(C)]),
    }


# ------------------------------------------------------------------ bootstrap
def bootstrap_indices(n, n_boot=2000, seed=0):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap(Y, S, thr, names, idx):
    """dict metric -> (n_boot, C) array, one row per resample in idx."""
    out = {k: np.empty((len(idx), Y.shape[1])) for k, _ in METRICS}
    for b, ii in enumerate(idx):
        m = per_class(Y[ii], S[ii], thr, names)
        for k in out:
            out[k][b] = m[k]
    return out


def nanmean_rows(a):
    with np.errstate(all="ignore"):
        cnt = np.isfinite(a).sum(1)
        return np.where(cnt > 0, np.nansum(a, 1) / np.maximum(cnt, 1), np.nan)


def ci(values, level=0.95):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return [float("nan"), float("nan")]
    a = (1 - level) / 2 * 100
    return [float(np.percentile(v, a)), float(np.percentile(v, 100 - a))]


def summarize(Y, S, thr, names, n_boot=2000, seed=0, idx=None):
    """Point estimates + 95 % CIs, JSON-ready:
    {metric: {"mean": x, "mean_ci": [lo, hi], "per_class": {e: x}, "per_class_ci": {e: [lo, hi]}}}"""
    point = per_class(Y, S, thr, names)
    idx = bootstrap_indices(len(Y), n_boot, seed) if idx is None else idx
    boot = bootstrap(Y, S, thr, names, idx)
    out = {}
    for k, _ in METRICS:
        out[k] = {
            "mean": float(np.nanmean(point[k])) if np.isfinite(point[k]).any() else float("nan"),
            "mean_ci": ci(nanmean_rows(boot[k])),
            "per_class": {n: float(v) for n, v in zip(names, point[k])},
            "per_class_ci": {n: ci(boot[k][:, i]) for i, n in enumerate(names)},
        }
    out["n_clips"] = int(len(Y))
    out["positives"] = {n: int(Y[:, i].sum()) for i, n in enumerate(names)}
    out.update(single_label_scores(Y, S))
    return out


def paired_differences(runs, names, ref=0, n_boot=2000, seed=0):
    """runs: list of (Y, S, thr) on the SAME clips in the same order. Returns rows
    {run, metric, diff, lo, hi, significant} for the difference of each mean metric to runs[ref]."""
    n = len(runs[0][0])
    idx = bootstrap_indices(n, n_boot, seed)
    boots = [bootstrap(Y, S, thr, names, idx) for Y, S, thr in runs]
    points = [per_class(Y, S, thr, names) for Y, S, thr in runs]
    rows = []
    for r in range(len(runs)):
        if r == ref:
            continue
        for k, label in METRICS:
            d = nanmean_rows(boots[r][k]) - nanmean_rows(boots[ref][k])
            lo, hi = ci(d)
            diff = float(np.nanmean(points[r][k]) - np.nanmean(points[ref][k]))
            rows.append(dict(metric=k, label=label, diff=diff, lo=lo, hi=hi,
                             significant=bool(np.isfinite(lo) and not (lo <= 0 <= hi)), run_index=r))
    return rows


def single_label_scores(Y, S):
    """Side metric: accuracy (WAR) and mean recall (UAR) on clips with exactly one emotion (argmax)."""
    one = Y.sum(1) == 1
    if not one.any():
        return {}
    y1, p1 = Y[one].argmax(1), S[one].argmax(1)
    rec = [float((p1[y1 == c] == c).mean()) for c in range(Y.shape[1]) if (y1 == c).any()]
    return {"single_label_clips": int(one.sum()), "single_label_war": float((y1 == p1).mean()),
            "single_label_uar": float(np.mean(rec))}


# ------------------------------------------------------------------ clustering
def purity(labels, clusters):
    total = 0
    for c in np.unique(clusters):
        total += np.bincount(labels[clusters == c]).max()
    return float(total / len(labels))


def clustering_scores(Z, clusters, Y):
    """Silhouette on all clips (in the clustering space); NMI / ARI / purity against the emotion on
    clips with exactly ONE emotion (multi-label and unlabelled clips are excluded; counts reported)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    clusters = np.asarray(clusters)
    out = {"n_clips": int(len(clusters)), "n_clusters": int(len(np.unique(clusters)))}
    ok = 1 < out["n_clusters"] < len(clusters)
    out["silhouette"] = float(silhouette_score(Z, clusters)) if ok else float("nan")
    one = Y.sum(1) == 1
    out["n_single_label"] = int(one.sum())
    out["n_excluded_multi_or_none"] = int((~one).sum())
    if one.sum() >= 2:
        lab = Y[one].argmax(1)
        out["nmi"] = float(normalized_mutual_info_score(lab, clusters[one]))
        out["ari"] = float(adjusted_rand_score(lab, clusters[one]))
        out["purity"] = purity(lab, clusters[one])
    return out
