"""Frame sampling, one rule for all tracks.

Every clip (and every patient-video window) is represented by N = 32 frames at uniform bin
centres: index_i = floor((i + 0.5) / N * n_frames). A model looks at 16 of them:

    view 0 (even frames) and view 1 (odd frames)  - evaluation averages both views; within each of
                                                    the 16 temporal bins these are the frames at
                                                    0.25 and 0.75, i.e. S2D's 2-window test
    training: one random frame of each pair       - temporal jitter (+ horizontal flip)
"""
import random

import cv2
import numpy as np


def uniform_indices(n_frames, n=32):
    idx = ((np.arange(n) + 0.5) / n * n_frames).astype(int)
    return np.clip(idx, 0, max(n_frames - 1, 0))


def view_indices(n_cache=32, clip_len=16, view=0):
    per = n_cache // clip_len
    return [i * per + (view % per) for i in range(clip_len)]


def train_indices(n_cache=32, clip_len=16, rng=random):
    per = n_cache // clip_len
    return [i * per + rng.randrange(per) for i in range(clip_len)]


def read_all_frames(path):
    """All frames of a (short) video as BGR arrays, plus fps."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, fps


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n, fps


def read_frames_at(path, indices):
    """{index: BGR frame} for the requested frame indices, one sequential pass (grab/retrieve)."""
    wanted = set(int(i) for i in indices)
    out, i = {}, 0
    cap = cv2.VideoCapture(str(path))
    last = max(wanted) if wanted else -1
    while i <= last:
        if i in wanted:
            ok, fr = cap.read()
            if not ok:
                break
            out[i] = fr
        elif not cap.grab():
            break
        i += 1
    cap.release()
    return out
