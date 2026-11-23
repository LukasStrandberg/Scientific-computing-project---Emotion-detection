"""Load config.yaml into a nested attribute dictionary and resolve paths.

    cfg = load_config(model="s2d", overrides={"patient": {"preset": "maedfer"}})
    cfg.faces.detector        -> "retinaface"
    cfg.path("results")       -> absolute Path under the data root

The data root is, in order: env EMO_ROOT, DRIVE_ROOT given to init(), the repository folder.
Smoke-test mode (cfg.smoke_test or env EMO_SMOKE=1) moves the root to <repo>/cache/_smoke and
applies the tiny settings from common/smoke.py.
"""
import copy
import json
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
MODELS = ("s2d", "maedfer", "clustering")


class Cfg(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def wrap(obj):
        if isinstance(obj, dict) and not isinstance(obj, Cfg):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def to_dict(self):
        return json.loads(json.dumps(self, default=str))

    def path(self, key):
        """Absolute path for cfg.paths[key] (relative paths are under the data root)."""
        p = Path(os.path.expanduser(str(self.paths[key])))
        return p if p.is_absolute() else Path(self.root) / p

    def resolve(self, rel):
        """Absolute path for any root-relative path string."""
        p = Path(os.path.expanduser(str(rel)))
        return p if p.is_absolute() else Path(self.root) / p


def deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def load_config(model=None, overrides=None, root=None, config_file=None):
    with open(config_file or REPO / "config.yaml") as f:
        raw = yaml.safe_load(f)
    deep_update(raw, overrides)
    model = os.environ.get("EMO_MODEL") or model   # EMO_MODEL: used by tools/run_smoke.py
    if model is not None and model not in MODELS:
        raise ValueError(f"MODEL must be one of {MODELS}, got {model!r}")
    raw["model"] = model
    raw["smoke_test"] = bool(raw.get("smoke_test")) or env_flag("EMO_SMOKE")
    if raw["smoke_test"]:
        from common.smoke import SMOKE_OVERRIDES
        deep_update(raw, SMOKE_OVERRIDES)
        root = REPO / "cache" / "_smoke"
    else:
        root = os.environ.get("EMO_ROOT") or root or REPO
    raw["root"] = str(Path(root).resolve())
    raw["repo"] = str(REPO)
    cfg = Cfg.wrap(raw)
    split_file = Path(cfg.paths.split_file)
    if raw["smoke_test"]:
        cfg.paths.split_file = str(Path(cfg.root) / "splits" / split_file.name)
    elif not split_file.is_absolute():
        cfg.paths.split_file = str(REPO / split_file)
    return cfg


def save_config(cfg, path, extra=None):
    d = cfg.to_dict()
    if extra:
        d.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=2))
