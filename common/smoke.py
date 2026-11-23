"""SMOKE_TEST mode: tiny synthetic data so every notebook runs top-to-bottom for every MODEL on a CPU.

Enabled by `smoke_test: true` in config.yaml or env EMO_SMOKE=1. The data root moves to
<repo>/cache/_smoke. Synthetic ElderReact-style clips are drawn ellipse "faces" whose colour and
mouth depend on the labels; one 8 s "patient" video. Face detection is replaced by a centred box,
MediaPipe by deterministic pseudo-blendshapes, and missing checkpoints by randomly initialised
models with the REAL architectures (so shapes, key names and Grad-CAM layers are exercised).
"""
import numpy as np

SMOKE_OVERRIDES = {
    "train": {"workers": 0, "max_steps": 2, "amp": False},
    "s2d": {"epochs": 1, "batch": 2, "accum": 1, "eval_batch": 2},
    "maedfer": {"epochs": 1, "batch": 2, "accum": 1, "eval_batch": 2},
    "clustering": {"ae_epochs": 20, "k_range": [2, 4]},
    "eval": {"n_boot": 50, "gradcam_n": 2, "n_examples": 3},
    "patient": {"gradcam_n": 2, "review_fraction": 0.2},
}
N_CLIPS = {"train": 12, "dev": 8, "test": 6}


def _write_video(path, frames, fps):
    import cv2
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        wr.write(f)
    wr.release()


def _face_frames(rng, y, n_frames, size=(160, 120)):
    import cv2
    w, h = size
    hue = int(30 * (np.argmax(y) if y.sum() else 6))
    colour = cv2.cvtColor(np.uint8([[[hue, 160, 200]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    frames = []
    for t in range(n_frames):
        img = (rng.normal(90, 12, (h, w, 3))).clip(0, 255).astype(np.uint8)
        cx, cy = w // 2 + int(4 * np.sin(t / 3)), h // 2
        cv2.ellipse(img, (cx, cy), (32, 42), 0, 0, 360, colour, -1)
        cv2.circle(img, (cx - 12, cy - 10), 4, (30, 30, 30), -1)
        cv2.circle(img, (cx + 12, cy - 10), 4, (30, 30, 30), -1)
        mouth = 3 + int(8 * y[3]) + int(4 * abs(np.sin(t / 2)))     # "happiness" opens the mouth
        cv2.ellipse(img, (cx, cy + 18), (12, mouth), 0, 0, 360, (40, 40, 160), -1)
        frames.append(img)
    return frames


def ensure_data(cfg):
    """Create the synthetic dataset under the smoke root (once)."""
    root = cfg.path("elderreact")
    if (root / "test_labels.txt").exists():
        return
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    for part, n in N_CLIPS.items():
        (root / f"ElderReact_{part}").mkdir(exist_ok=True)
        Y = (rng.random((n, 6)) < 0.35).astype(int)
        if part == "train":
            Y[np.arange(6), np.arange(6)] = 1          # every emotion occurs in training
        with open(root / f"{part}_labels.txt", "w") as f:
            for i, y in enumerate(Y):
                name = f"smoke_{part}_{i:02d}"
                _write_video(root / f"ElderReact_{part}" / f"{name}.mp4", _face_frames(rng, y, 20), 10)
                f.write(f"{name}.mp4 {' '.join(map(str, y))} f 2.0\n")
    pv = cfg.path("patient_videos")
    pv.mkdir(parents=True, exist_ok=True)
    frames = []
    for seg in range(4):                                # 4 x 2 s with different "expressions"
        y = np.zeros(6, int)
        y[seg % 6] = 1
        frames += _face_frames(rng, y, 20)
    _write_video(pv / "smoke_patient.mp4", frames, 10)
    print(f"SMOKE_TEST: synthetic data written to {cfg.root}")
