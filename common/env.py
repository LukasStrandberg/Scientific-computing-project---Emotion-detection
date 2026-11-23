"""Environment handling: Colab vs local, Drive, code checkout, packages, weights, device, seeds.

00_setup.ipynb calls setup(); every other notebook starts with init(MODEL).

On Colab every notebook has its own runtime, so setup() writes a small bootstrap script to
MyDrive (BOOTSTRAP) that the other notebooks execute: it re-clones the code if needed, installs the
requirements once per runtime and sets the data root (EMO_ROOT) to your DRIVE_ROOT.
"""
import hashlib
import os
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

from common.config import REPO, load_config

COLAB_REPO = Path("/content/emotion-detection-unified")
BOOTSTRAP = Path("/content/drive/MyDrive/emotion_detection_bootstrap.py")
S2D_COMMIT = "e1b5e5dfb57ab15f70103c933b346bfa62943159"
SMALL_DOWNLOADS = {   # (url, path relative to repo or data root)
    "mobilefacenet": (f"https://raw.githubusercontent.com/MSA-LMC/S2D/{S2D_COMMIT}/landmark/mobilefacenet_model_best.pth.tar",
                      "repo:third_party/s2d/landmark/mobilefacenet_model_best.pth.tar"),
    "face_landmarker": ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                        "face_landmarker.task", "cfg:clustering.landmarker"),
}


def in_colab():
    return "google.colab" in sys.modules


def sh(cmd, check=True):
    print("$", " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else cmd)
    return subprocess.run(cmd, check=check, shell=isinstance(cmd, str))


def get_device():
    import torch
    if os.environ.get("EMO_DEVICE"):
        return os.environ["EMO_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ setup (00_setup)
def _bootstrap_text(drive_root, repo_url, branch):
    return f'''# Written by 00_setup.ipynb - executed at the top of every notebook on Colab.
import os, subprocess, sys
DRIVE_ROOT, REPO_URL, BRANCH = {str(drive_root)!r}, {repo_url!r}, {branch!r}
REPO = "{COLAB_REPO}"
if not os.path.isdir(REPO + "/common"):
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO_URL, REPO], check=True)
if not os.path.exists("/content/.emo_requirements_installed"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", REPO + "/requirements.txt"], check=True)
    open("/content/.emo_requirements_installed", "w").close()
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ["EMO_ROOT"] = DRIVE_ROOT
'''


def setup(drive_root=None, repo_url=None, branch=None, install=True):
    """Prepare the environment. Colab: Drive, code, packages, bootstrap. Local: checks only."""
    if in_colab():
        if not drive_root:
            raise SystemExit("Set DRIVE_ROOT in the first cell, e.g. DRIVE_ROOT = '/content/drive/MyDrive/emotion' "
                             "(a folder on your Drive that holds data/, checkpoints/, cache/, results/).")
        from google.colab import drive
        drive.mount("/content/drive")
        Path(drive_root).mkdir(parents=True, exist_ok=True)
        if repo_url and (COLAB_REPO / ".git").exists():
            sh(["git", "-C", str(COLAB_REPO), "pull", "-q"], check=False)
        if install:
            sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")])
            Path("/content/.emo_requirements_installed").touch()
        BOOTSTRAP.write_text(_bootstrap_text(drive_root, repo_url or "", branch or ""))
        print(f"Bootstrap for the other notebooks written to {BOOTSTRAP}")
        os.environ["EMO_ROOT"] = str(drive_root)
    else:
        if drive_root:
            os.environ["EMO_ROOT"] = str(drive_root)
        in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        print(f"Python {sys.version.split()[0]} at {sys.executable} ({'venv' if in_venv else 'NOT a venv'})")
        if not in_venv:
            print("  Hint: create the venv (README, 'Setup') and select it as the notebook kernel.")
    cfg = load_config()
    for key in ("data", "checkpoints", "cache", "results"):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
    report_packages()
    download_small_weights(cfg)
    if cfg.smoke_test:
        from common import smoke
        smoke.ensure_data(cfg)
    print_weights_table(cfg)
    return cfg


def report_packages():
    import importlib
    rows = []
    for name in ("torch", "timm", "numpy", "cv2", "mediapipe", "batch_face", "ultralytics", "sklearn", "matplotlib"):
        try:
            mod = importlib.import_module(name)
            rows.append(f"{name} {getattr(mod, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            rows.append(f"{name} MISSING ({type(e).__name__})")
    print("Packages:", " | ".join(rows))
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
    else:
        print("No CUDA GPU found - preprocessing and training will be slow (fine for SMOKE_TEST).")
    import timm
    if timm.__version__ != "0.6.13":
        print(f"WARNING: timm {timm.__version__} installed, the code is tested with 0.6.13 (pip install timm==0.6.13)")


def _target(cfg, spec):
    kind, rel = spec.split(":", 1)
    if kind == "repo":
        return REPO / rel
    node = cfg
    for part in rel.split("."):
        node = node[part]
    return cfg.resolve(node)


def download_small_weights(cfg):
    """MobileFaceNet (12 MB) and the MediaPipe face landmarker (4 MB). Large checkpoints are manual."""
    for name, (url, spec) in SMALL_DOWNLOADS.items():
        dst = _target(cfg, spec)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"downloading {name} -> {dst}")
            urllib.request.urlretrieve(url, dst)
        except Exception as e:  # noqa: BLE001
            print(f"  could not download {name} ({e}); place it manually at {dst}")


def weights_table(cfg):
    """(name, path, needed for, present, where to get it)."""
    return [
        ("S2D MAFW fold-1 checkpoint", cfg.resolve(cfg.s2d.mafw_ckpt), "s2d",
         "huggingface.co/cyinen/S2D -> MAFW/without_oversampler/split01_checkpoint-best.pth"),
        ("MobileFaceNet landmarks", _target(cfg, SMALL_DOWNLOADS["mobilefacenet"][1]), "s2d", "auto-download (S2D repo)"),
        ("MAE-DFER DFEW checkpoint", cfg.resolve(cfg.maedfer.dfew_ckpt), "maedfer",
         "github.com/sunlicai/MAE-DFER README -> DFEW fold 1 checkpoint (Google Drive)"),
        ("MediaPipe face landmarker", cfg.resolve(cfg.clustering.landmarker), "clustering", "auto-download (Google)"),
        ("YOLOv8n-face (optional)", cfg.resolve(cfg.faces.yolo_weights), "faces.detector: yolo",
         "e.g. github.com/akanametov/yolo-face releases -> yolov8n-face.pt"),
    ]


def print_weights_table(cfg):
    print("\nWeights (relative to the data root unless shown otherwise):")
    for name, path, needed, src in weights_table(cfg):
        state = "OK     " if Path(path).exists() else "MISSING"
        print(f"  {state} {name:28s} [{needed}]  {path}\n          source: {src}")
    if cfg.smoke_test:
        print("  (SMOKE_TEST: missing checkpoints are replaced by randomly initialised models)")


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ init (01-05)
def init(model=None, **overrides):
    """Load the config for one notebook; overrides are nested dicts or dotted keys."""
    if in_colab() and not os.environ.get("EMO_ROOT"):
        raise SystemExit("Run 00_setup.ipynb first (it writes the Colab bootstrap with your DRIVE_ROOT).")
    nested = {}
    for key, val in overrides.items():
        node = nested
        parts = key.split("__")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    cfg = load_config(model, nested)
    for key in ("data", "checkpoints", "cache", "results"):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
    if cfg.smoke_test:
        from common import smoke
        smoke.ensure_data(cfg)
    seed_everything(cfg.seed)
    from common import plotting
    plotting.use_style(cfg)
    print(f"MODEL = {cfg.model!r} | data root: {cfg.root}" + ("  [SMOKE_TEST]" if cfg.smoke_test else "")
          + f" | device: {get_device()}")
    return cfg
