"""The shared face-crop cache: 32 uniformly sampled, face-cropped 224x224 frames per clip.

One file per clip, <cache_dir>/<clip>.npz with
    jpeg + offsets   (format jpeg: JPEG bytes of each frame, concatenated)  or  frames (format npy)
    has_face (32,)   detection found on that frame (others reuse the nearest box)
    frame_idx (32,)  source frame numbers;  n_video_frames, fps, status ("ok" | "no_face")
A report <cache_dir>/_report.csv lists every clip. Building is resumable (finished clips are
skipped). On Colab the cache lives on the local disk and is kept on Drive as ONE tar file.
"""
import csv
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np

from common import sampling
from common.env import in_colab

REPORT_FIELDS = ["clip", "n_video_frames", "frames_without_face", "status"]


def cache_tag(cfg):
    f = cfg.faces
    tag = f"faces{cfg.cache.size}_{f.detector}_{f.style}_f{cfg.cache.n_frames}"
    if f.margin:
        tag += f"_m{int(round(100 * f.margin)):02d}"
    if f.per_clip_box:
        tag += "_clipbox"
    return tag + ("_jpg" if cfg.cache.format == "jpeg" else "")


def cache_dir(cfg):
    if in_colab() and not cfg.smoke_test:
        return Path(cfg.paths.colab_local_cache) / cache_tag(cfg)
    return cfg.path("cache") / cache_tag(cfg)


def clip_path(cfg, clip):
    return cache_dir(cfg) / f"{clip}.npz"


# ------------------------------------------------------------------ file format
def save_clip(path, crops_rgb, has_face, meta, fmt="jpeg", quality=95):
    path = Path(path)
    arrays = dict(has_face=np.asarray(has_face, bool), **{k: np.asarray(v) for k, v in meta.items()})
    if fmt == "jpeg":
        blobs = [cv2.imencode(".jpg", cv2.cvtColor(c, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
                 for c in crops_rgb]
        arrays["offsets"] = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
        arrays["jpeg"] = np.concatenate(blobs).astype(np.uint8)
    else:
        arrays["frames"] = np.asarray(crops_rgb, np.uint8)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def load_clip(path, indices=None):
    """uint8 RGB frames (k, S, S, 3) for the given cached-frame indices (all if None)."""
    with np.load(path) as d:
        if "frames" in d:
            fr = d["frames"]
            return fr if indices is None else fr[list(indices)]
        off, buf = d["offsets"], d["jpeg"]
        idx = range(len(off) - 1) if indices is None else indices
        return np.stack([cv2.cvtColor(cv2.imdecode(buf[off[i]:off[i + 1]], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                         for i in idx])


def load_meta(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files if k not in ("frames", "jpeg", "offsets")}


# ------------------------------------------------------------------ building
def read_report(cfg):
    p = cache_dir(cfg) / "_report.csv"
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        return {r["clip"]: r for r in csv.DictReader(f)}


def build_cache(cfg, df, detector, redo=False, log_every=50):
    """Crop every clip of df (columns video, path). Returns the report as a list of dicts."""
    out = cache_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    report = read_report(cfg)
    todo = [(c, p) for c, p in zip(df.video, df.path) if redo or not clip_path(cfg, c).exists()]
    print(f"cache {out}: {len(df) - len(todo)} clips already cached, {len(todo)} to build")
    t0 = time.time()
    for n, (clip, vid) in enumerate(todo, 1):
        frames, fps = sampling.read_all_frames(vid)
        if not frames:
            report[clip] = dict(clip=clip, n_video_frames=0, frames_without_face=cfg.cache.n_frames, status="unreadable")
            print(f"  {clip}: unreadable video {vid}")
            continue
        idx = sampling.uniform_indices(len(frames), cfg.cache.n_frames)
        crops, has_face, boxes, status = crop_frames_for(cfg, [frames[i] for i in idx], detector)
        save_clip(clip_path(cfg, clip), crops, has_face,
                  dict(frame_idx=idx, n_video_frames=len(frames), fps=fps, status=status, boxes=boxes),
                  cfg.cache.format, cfg.cache.jpeg_quality)
        report[clip] = dict(clip=clip, n_video_frames=len(frames), frames_without_face=int((~has_face).sum()),
                            status=status)
        if n % log_every == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  [{n}/{len(todo)}] {el / 60:.1f} min, ~{el / n * (len(todo) - n) / 60:.1f} min left")
            _write_report(cfg, report)
    _write_report(cfg, report)
    return list(report.values())


def crop_frames_for(cfg, frames, detector):
    from common.faces import crop_frames
    return crop_frames(frames, detector, cfg)


def _write_report(cfg, report):
    with open(cache_dir(cfg) / "_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(report.values(), key=lambda r: r["clip"]))


def missing_clips(cfg, clips):
    return [c for c in clips if not clip_path(cfg, c).exists()]


# ------------------------------------------------------------------ Drive tar (Colab)
def tar_path(cfg):
    return cfg.path("cache") / f"{cache_tag(cfg)}.tar"


def uses_tar(cfg):
    return in_colab() and cfg.cache.tar_to_drive and not cfg.smoke_test


def load_from_drive(cfg):
    """Colab: extract <root>/cache/<tag>.tar to the local disk (existing files are kept)."""
    if not uses_tar(cfg):
        return False
    tp, out = tar_path(cfg), cache_dir(cfg)
    if not tp.exists():
        print(f"no cache tar at {tp} - the cache will be built from the videos")
        return False
    if (out / "_report.csv").exists():
        return True
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"extracting {tp} ({tp.stat().st_size / 1e9:.2f} GB) ...")
    with tarfile.open(tp) as tf:
        try:
            tf.extractall(out.parent, filter="data")
        except TypeError:   # Python without tarfile extraction filters
            tf.extractall(out.parent)
    print(f"extracted in {time.time() - t0:.0f} s")
    return True


def save_to_drive(cfg):
    """Colab: write the cache folder as one uncompressed tar to Drive (tmp file, then rename)."""
    if not uses_tar(cfg):
        return
    tp, src = tar_path(cfg), cache_dir(cfg)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tmp, t0 = tp.with_suffix(".tar.tmp"), time.time()
    with tarfile.open(tmp, "w") as tf:
        tf.add(src, arcname=src.name)
    tmp.replace(tp)
    print(f"saved cache tar {tp} ({tp.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.0f} s")


def ensure_cache(cfg, clips):
    """Make sure the cache for `clips` is available locally (extracting the Drive tar if needed)."""
    load_from_drive(cfg)
    miss = missing_clips(cfg, clips)
    if miss:
        raise FileNotFoundError(f"{len(miss)} clips are not in the face cache {cache_dir(cfg)} "
                                f"(e.g. {miss[:3]}) - run 01_preprocess.ipynb first.")
