"""The output schema shared by every run. 05_compare reads only what is written here.

results/<model>/<run>/
    config.json       resolved settings (+ checkpoint hashes, split file)
    metrics.json      {"val": summary, "test": summary[, "clustering": {...}]}, see metrics.summarize
    thresholds.json   per-emotion thresholds tuned on the validation split
    predictions.csv   video, split, true_<emotion>..., prob_<emotion>... [, cluster]
    features.npz      features (N, D), video, split, labels (N, 6), probs (N, 6) [, cluster]
    log.csv           training log (supervised fine-tuning; clustering: autoencoder loss)
    figures/          PDF + PNG
    patients/<video>_timeline.csv   start_s, end_s, centre_s, face_share, prob_<emotion>... [, cluster], predicted
    patients/<video>_segments.csv, <video>_summary.json, patients/figures/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run_dir(cfg, model, run):
    return cfg.path("results") / model / run


def prediction_frame(videos, splits, Y, P, emotions, clusters=None):
    df = pd.DataFrame({"video": list(videos), "split": list(splits)})
    for i, e in enumerate(emotions):
        df[f"true_{e}"] = Y[:, i].astype(int)
    for i, e in enumerate(emotions):
        df[f"prob_{e}"] = P[:, i].astype(float)   # full precision: thresholds must match exactly
    if clusters is not None:
        df["cluster"] = np.asarray(clusters).astype(int)
    return df


def write_run(out, predictions, metrics, thresholds, features, config):
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    np.savez(out / "features.npz", **features)
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))


def read_predictions(out, emotions):
    """(df, Y, P) from a run folder or a predictions.csv path."""
    p = Path(out)
    p = p / "predictions.csv" if p.is_dir() else p
    df = pd.read_csv(p)
    df["video"] = df["video"].astype(str).map(lambda v: Path(v).stem if v.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")) else v)
    Y = df[[f"true_{e}" for e in emotions]].to_numpy(dtype=float)
    P = df[[f"prob_{e}" for e in emotions]].to_numpy(dtype=float)
    return df, Y, P


def read_run(out, emotions):
    out = Path(out)
    df, Y, P = read_predictions(out, emotions)
    run = {"dir": out, "predictions": df, "Y": Y, "P": P,
           "metrics": json.loads((out / "metrics.json").read_text()),
           "thresholds": json.loads((out / "thresholds.json").read_text()),
           "config": json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}}
    if (out / "features.npz").exists():
        with np.load(out / "features.npz", allow_pickle=False) as f:
            run["features"] = {k: f[k] for k in f.files}
    return run


def find_runs(cfg):
    """All (model, run) folders under results/ that follow the schema."""
    root = cfg.path("results")
    return sorted((p.parent.parent.name, p.parent.name) for p in root.glob("*/*/predictions.csv"))


def read_log(out):
    p = Path(out) / "log.csv"
    return pd.read_csv(p) if p.exists() else None
