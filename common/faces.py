"""Face detection and cropping, shared by all tracks.

Default (config `faces`): RetinaFace-ResNet50 via batch-face, one detection per sampled frame, the
largest face, frame rotated so the eyes are level, "deepface" crop (tight face box resized
keeping its aspect ratio, padded with black to a square). Frames without a face reuse the box of
the nearest frame that had one. A clip without any face falls back to a centred square of the
whole frame and is flagged (`status = no_face`), so every track is scored on the same clips.

Alternative: YOLOv8n-face (`faces.detector: yolo`, boxes only, no eye levelling), optionally with
one box for the whole clip (`faces.per_clip_box: true`, the MAE-DFER track's earlier protocol).
"""
import math

import cv2
import numpy as np


class RetinaFaceDetector:
    def __init__(self, cfg, device):
        from batch_face import RetinaFace
        use_gpu = device == "cuda"
        self.net = RetinaFace(gpu_id=0 if use_gpu else -1, network=cfg.faces.retinaface_network, fp16=use_gpu)
        self.f = cfg.faces

    def __call__(self, frames_bgr):
        """list of (box[4], kps[5,2]) or None per frame (largest face above min_score)."""
        out = []
        for i in range(0, len(frames_bgr), self.f.batch):
            res = self.net(frames_bgr[i:i + self.f.batch], cv=True, threshold=self.f.min_score,
                           max_size=self.f.max_size, batch_size=self.f.batch)
            out += [largest_face([(np.asarray(b), np.asarray(k), s) for b, k, s in r], self.f.min_score) for r in res]
        return out


class YoloDetector:
    def __init__(self, cfg, device):
        from ultralytics import YOLO
        self.model = YOLO(str(cfg.resolve(cfg.faces.yolo_weights)))
        self.f, self.device = cfg.faces, (0 if device == "cuda" else "cpu")

    def __call__(self, frames_bgr):
        out = []
        for i in range(0, len(frames_bgr), self.f.batch):
            res = self.model.predict(source=list(frames_bgr[i:i + self.f.batch]), conf=self.f.yolo_conf,
                                     imgsz=self.f.yolo_imgsz, device=self.device, verbose=False)
            for r in res:
                boxes = [] if r.boxes is None else r.boxes.xyxy.cpu().numpy()
                out.append(largest_face([(b, None, 1.0) for b in boxes], 0.0))
        return out


class StubDetector:
    """SMOKE_TEST only: a centred box covering the middle of the frame, eyes level."""

    def __call__(self, frames_bgr):
        out = []
        for fr in frames_bgr:
            h, w = fr.shape[:2]
            box = np.array([0.25 * w, 0.2 * h, 0.75 * w, 0.8 * h], dtype=np.float32)
            kps = np.array([[0.4 * w, 0.4 * h], [0.6 * w, 0.4 * h], [0.5 * w, 0.5 * h],
                            [0.42 * w, 0.65 * h], [0.58 * w, 0.65 * h]], dtype=np.float32)
            out.append((box, kps))
        return out


def make_detector(cfg, device, kind=None):
    kind = kind or cfg.faces.detector
    if cfg.smoke_test:
        return StubDetector()
    if kind == "retinaface":
        return RetinaFaceDetector(cfg, device)
    if kind == "yolo":
        return YoloDetector(cfg, device)
    raise ValueError(f"unknown faces.detector {kind!r}")


def largest_face(faces, min_score):
    """faces: [(box[x1,y1,x2,y2], kps or None, score)] -> (box, kps) of the largest, or None."""
    faces = [f for f in faces if f[2] >= min_score]
    if not faces:
        return None
    box, kps, _ = max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
    return np.asarray(box, dtype=np.float32), (None if kps is None else np.asarray(kps, dtype=np.float32))


def fill_missing(dets):
    """Replace None entries with the nearest detection in time; None if there is none at all."""
    have = [i for i, d in enumerate(dets) if d is not None]
    if not have:
        return None
    return [d if d is not None else dets[min(have, key=lambda k: abs(k - i))] for i, d in enumerate(dets)]


def align_crop(frame, box, kps, size, margin=0.0, style="deepface"):
    """Rotate so the eyes are level (if landmarks are given), cut out the face, resize to size x size.

    style="square":   square region of the image around the face
    style="deepface": tight face box, resized keeping its aspect ratio, padded with black
                      (mimics deepface's extract_faces, which the S2D authors used)
    """
    angle = 0.0
    if kps is not None:
        (lx, ly), (rx, ry) = kps[0], kps[1]
        angle = math.degrees(math.atan2(ry - ly, rx - lx))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = (box[2] - box[0]) * (1 + margin), (box[3] - box[1]) * (1 + margin)
    scale = size / max(w, h, 1e-6)
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, scale)
    M[0, 2] += size / 2 - cx
    M[1, 2] += size / 2 - cy
    out = cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if style == "deepface":
        pw, ph = int(round(w * scale)), int(round(h * scale))
        mask = np.zeros((size, size), dtype=bool)
        x0, y0 = (size - pw) // 2, (size - ph) // 2
        mask[max(y0, 0):y0 + ph, max(x0, 0):x0 + pw] = True
        out[~mask] = 0
    return out


def centre_box(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    return np.array([(w - s) / 2, (h - s) / 2, (w + s) / 2, (h + s) / 2], dtype=np.float32)


def crop_frames(frames_bgr, detector, cfg):
    """Detect + crop a list of BGR frames.

    Returns (crops RGB uint8 (N,S,S,3), has_face bool (N,), boxes (N,4), status)."""
    f, size = cfg.faces, cfg.cache.size
    n = len(frames_bgr)
    if f.per_clip_box:   # one box for the clip: first detection among the middle, 1/4 and 3/4 frames
        det = None
        for j in (n // 2, n // 4, 3 * n // 4):
            det = detector([frames_bgr[j]])[0]
            if det is not None:
                break
        raw = [None] * n if det is None else [(det[0], None)] * n
        has_face = np.full(n, det is not None)
    else:
        raw = detector(frames_bgr)
        has_face = np.array([d is not None for d in raw])
    dets = fill_missing(raw)
    status = "ok"
    if dets is None:
        status = "no_face"
        dets = [(centre_box(fr), None) for fr in frames_bgr]
    style = f.style if status == "ok" else "square"
    crops = np.stack([cv2.cvtColor(align_crop(fr, b, k, size, f.margin, style), cv2.COLOR_BGR2RGB)
                      for fr, (b, k) in zip(frames_bgr, dets)])
    boxes = np.stack([b for b, _ in dets]).astype(np.float32)
    return crops, has_face, boxes, status
