"""The run schema is written by outputs.write_run and read back by 05_compare (stages.compare)."""
import json

import numpy as np
import pytest

from common import metrics, outputs, stages
from common.config import load_config

EMO = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def _fake_run(cfg, model, run, Yv, Yt, rng, clusters=False):
    Pv, Pt = rng.random(Yv.shape), rng.random(Yt.shape)
    thr = metrics.tune_thresholds(Yv, Pv, EMO)
    m = {"val": metrics.summarize(Yv, Pv, thr, EMO, n_boot=20), "test": metrics.summarize(Yt, Pt, thr, EMO, n_boot=20)}
    vids = [f"v{i}" for i in range(len(Yv))] + [f"t{i}" for i in range(len(Yt))]
    splits = ["val"] * len(Yv) + ["test"] * len(Yt)
    cl = rng.integers(0, 3, len(vids)) if clusters else None
    pred = outputs.prediction_frame(vids, splits, np.vstack([Yv, Yt]), np.vstack([Pv, Pt]), EMO, cl)
    feats = {"features": rng.normal(size=(len(vids), 8)).astype(np.float32), "video": np.asarray(vids, dtype=str),
             "split": np.asarray(splits, dtype=str), "labels": np.vstack([Yv, Yt]), "probs": np.vstack([Pv, Pt])}
    outputs.write_run(outputs.run_dir(cfg, model, run), pred, m, thr, feats, {"model": model, "run": run})
    return pred


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.delenv("EMO_ROOT", raising=False)
    c = load_config(root=tmp_path, overrides={"eval": {"n_boot": 30}})
    from common import plotting
    plotting.use_style(c)
    return c


def test_schema_roundtrip_and_compare(cfg):
    rng = np.random.default_rng(0)
    Yv = (rng.random((30, 6)) < 0.4).astype(float)
    Yt = (rng.random((40, 6)) < 0.4).astype(float)
    pred = _fake_run(cfg, "s2d", "finetuned", Yv, Yt, rng)
    _fake_run(cfg, "maedfer", "finetuned", Yv, Yt, rng)
    _fake_run(cfg, "clustering", "ae_kmeans", Yv, Yt, rng, clusters=True)

    d = outputs.run_dir(cfg, "s2d", "finetuned")
    for f in ("config.json", "metrics.json", "thresholds.json", "predictions.csv", "features.npz"):
        assert (d / f).exists()
    assert list(pred.columns[:8]) == ["video", "split"] + [f"true_{e}" for e in EMO]
    r = outputs.read_run(d, EMO)
    assert np.array_equal(r["P"], pred[[f"prob_{e}" for e in EMO]].to_numpy())   # full precision
    assert set(json.loads((d / "metrics.json").read_text())) == {"val", "test"}
    assert "cluster" in outputs.read_run(outputs.run_dir(cfg, "clustering", "ae_kmeans"), EMO)["predictions"]

    assert outputs.find_runs(cfg) == [("clustering", "ae_kmeans"), ("maedfer", "finetuned"), ("s2d", "finetuned")]
    table, diffs = stages.compare(cfg, reference="s2d/finetuned")
    assert len(table) == 3
    assert set(diffs.run) == {"maedfer/finetuned", "clustering/ae_kmeans"}
    out = cfg.path("results") / "compare"
    assert (out / "test_metrics.csv").exists() and (out / "figures" / "per_class_auc.pdf").exists()
    assert (out / "figures" / "per_class_auc.png").exists()


def test_compare_rejects_different_test_sets(cfg):
    rng = np.random.default_rng(1)
    Yv = (rng.random((20, 6)) < 0.4).astype(float)
    _fake_run(cfg, "s2d", "finetuned", Yv, (rng.random((30, 6)) < 0.4).astype(float), rng)
    _fake_run(cfg, "maedfer", "finetuned", Yv, (rng.random((25, 6)) < 0.4).astype(float), rng)
    with pytest.raises(ValueError):
        stages.compare(cfg)
