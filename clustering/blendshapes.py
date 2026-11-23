"""MediaPipe FaceLandmarker blendshapes (52 facial action scores in [0, 1]) per frame, and the
per-clip / per-window vector [mean, max, std] over the frames where a face was found (156-d).

Frames are the same 32 face crops the supervised tracks use (shared cache). `clustering.source:
full_frame` instead reads the whole video frames at the same frame indices.
"""
import numpy as np

BLENDSHAPES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight"]
FEATURE_NAMES = [f"{s}_{b}" for s in ("mean", "max", "std") for b in BLENDSHAPES]


class BlendshapeExtractor:
    def __init__(self, cfg):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        path = cfg.resolve(cfg.clustering.landmarker)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run 00_setup.ipynb (downloads face_landmarker.task)")
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(path)),
                                            running_mode=vision.RunningMode.IMAGE, num_faces=1,
                                            output_face_blendshapes=True)
        self.mp, self.lm = mp, vision.FaceLandmarker.create_from_options(opts)

    def __call__(self, frames_rgb):
        """(N, H, W, 3) uint8 RGB -> (N, 52), NaN rows where no face was found."""
        out = np.full((len(frames_rgb), len(BLENDSHAPES)), np.nan, np.float32)
        for i, fr in enumerate(frames_rgb):
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr))
            res = self.lm.detect(img)
            if res.face_blendshapes:
                scores = {c.category_name: c.score for c in res.face_blendshapes[0]}
                out[i] = [scores.get(b, np.nan) for b in BLENDSHAPES]
        return out


class StubExtractor:
    """SMOKE_TEST only: deterministic pseudo-blendshapes from a 4x4 colour grid of the frame, so that
    synthetic clips with different 'faces' end up in different clusters."""

    def __init__(self, seed=0):
        self.W = np.random.default_rng(seed).normal(size=(48, len(BLENDSHAPES))).astype(np.float32)

    def __call__(self, frames_rgb):
        f = np.asarray(frames_rgb, np.float32) / 255.0
        n, h, w, _ = f.shape
        grid = f[:, : h // 4 * 4, : w // 4 * 4].reshape(n, 4, h // 4, 4, w // 4, 3).mean((2, 4)).reshape(n, 48)
        return (1 / (1 + np.exp(-(grid - 0.3) @ self.W))).astype(np.float32)


def make_extractor(cfg):
    return StubExtractor() if cfg.smoke_test else BlendshapeExtractor(cfg)


def aggregate(per_frame):
    """(N, 52) with NaN rows -> (156,) [mean, max, std]; NaN vector if no frame has a face."""
    ok = np.isfinite(per_frame).all(1)
    if not ok.any():
        return np.full(3 * per_frame.shape[1], np.nan, np.float32)
    x = per_frame[ok]
    std = x.std(0, ddof=1) if len(x) > 1 else np.zeros(x.shape[1])
    return np.concatenate([x.mean(0), x.max(0), std]).astype(np.float32)
