import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common import metrics

EMO = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    Y = (rng.random((120, 6)) < 0.3).astype(float)
    S = np.clip(0.3 * Y + rng.random((120, 6)) * 0.7, 0, 1)
    S[:10] = np.round(S[:10], 1)             # include ties
    return Y, S


def test_auc_ap_f1_match_sklearn(data):
    Y, S = data
    for i in range(6):
        assert metrics.auc(Y[:, i], S[:, i]) == pytest.approx(roc_auc_score(Y[:, i], S[:, i]))
        assert metrics.average_precision(Y[:, i], S[:, i]) == pytest.approx(average_precision_score(Y[:, i], S[:, i]))
        assert metrics.f1(Y[:, i], S[:, i] >= 0.5) == pytest.approx(f1_score(Y[:, i], S[:, i] >= 0.5))


def test_undefined_cases():
    assert np.isnan(metrics.auc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert np.isnan(metrics.average_precision([0, 0], [0.1, 0.2]))
    assert metrics.f1([0, 0], [0, 0]) == 0.0


def test_threshold_tuning_maximises_f1(data):
    Y, S = data
    thr = metrics.tune_thresholds(Y, S, EMO)
    for i, e in enumerate(EMO):
        best = metrics.f1(Y[:, i], S[:, i] >= thr[e])
        assert all(best >= metrics.f1(Y[:, i], S[:, i] >= t) - 1e-12 for t in np.linspace(0, 1, 101))


def test_bootstrap_shapes_and_ci(data):
    Y, S = data
    thr = {e: 0.5 for e in EMO}
    idx = metrics.bootstrap_indices(len(Y), 200, 0)
    assert idx.shape == (200, len(Y))
    boot = metrics.bootstrap(Y, S, thr, EMO, idx)
    assert set(boot) == {k for k, _ in metrics.METRICS}
    assert all(v.shape == (200, 6) for v in boot.values())
    s = metrics.summarize(Y, S, thr, EMO, n_boot=200)
    for k, _ in metrics.METRICS:
        lo, hi = s[k]["mean_ci"]
        assert lo <= s[k]["mean"] <= hi or np.isclose(lo, s[k]["mean"]) or np.isclose(hi, s[k]["mean"])
        assert set(s[k]["per_class"]) == set(EMO)
    assert s["n_clips"] == len(Y)


def test_paired_differences(data):
    Y, S = data
    thr = {e: 0.5 for e in EMO}
    rows = metrics.paired_differences([(Y, S, thr), (Y, S, thr), (Y, 1 - S, thr)], EMO, n_boot=200)
    same = [r for r in rows if r["run_index"] == 1]
    assert all(r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0 and not r["significant"] for r in same)
    worse = [r for r in rows if r["run_index"] == 2 and r["metric"] == "auc"][0]
    assert worse["diff"] < 0 and worse["significant"]
    assert len(rows) == 2 * len(metrics.METRICS)


def test_clustering_scores():
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rng = np.random.default_rng(1)
    Z = np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))])
    clusters = np.array([0] * 20 + [1] * 20)
    Y = np.zeros((40, 6))
    Y[:20, 0] = 1
    Y[20:30, 3] = 1
    Y[30:, [1, 2]] = 1                         # multi-label: excluded from NMI / ARI / purity
    s = metrics.clustering_scores(Z, clusters, Y)
    assert s["n_single_label"] == 30 and s["n_excluded_multi_or_none"] == 10
    lab = Y[:30].argmax(1)
    assert s["nmi"] == pytest.approx(normalized_mutual_info_score(lab, clusters[:30]))
    assert s["ari"] == pytest.approx(adjusted_rand_score(lab, clusters[:30]))
    assert s["purity"] == 1.0 and s["silhouette"] > 0.9
