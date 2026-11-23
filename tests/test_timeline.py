import numpy as np
import pytest

from common import timeline
from common.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.mark.parametrize("preset,duration,expected", [
    ("s2d", 60.0, (640 - 32) // 11 + 1),       # step 3/32 s, stride 11 steps = 1.031 s
    ("s2d", 3.0, 1),                            # exactly one window
    ("s2d", 2.9, 0),                            # shorter than a window
    ("maedfer", 10.0, (320 - 32) // 26 + 1),   # step 1/32 s, stride 26 steps = 0.8125 s
])
def test_window_counts(cfg, preset, duration, expected):
    ws = timeline.window_settings(cfg, preset)
    assert timeline.expected_windows(duration, ws) == expected


def test_presets(cfg):
    s2d, mae = timeline.window_settings(cfg, "s2d"), timeline.window_settings(cfg, "maedfer")
    assert (s2d["window"], s2d["smooth"]) == (3.0, 5) and s2d["stride"] == pytest.approx(1.03125)
    assert (mae["window"], mae["smooth"]) == (1.0, 1) and mae["stride"] == pytest.approx(0.8125)


def test_smoothing():
    x = np.tile(np.array([0.2, 0.8]), (7, 1)).astype(float)
    assert np.allclose(timeline.smooth(x, 5), x)                  # constant stays constant
    y = np.arange(7, dtype=float)[:, None]
    assert timeline.smooth(y, 3)[3, 0] == pytest.approx(3.0)
    assert timeline.smooth(y, 3)[0, 0] == pytest.approx(0.5)      # window truncated at the edge
    y[3] = np.nan
    s = timeline.smooth(y, 3)
    assert np.isnan(s[3, 0]) and s[2, 0] == pytest.approx(1.5)    # no-face window ignored
    assert np.array_equal(timeline.smooth(y, 1), y, equal_nan=True)


def test_dominant_and_segments():
    names = ["a", "b"]
    sc = np.array([[0.9, 0.1], [0.2, 0.3], [np.nan, np.nan], [0.1, 0.9]])
    assert timeline.dominant(sc, names, 0.5) == ["a", timeline.NO_CLEAR, timeline.NO_FACE, "b"]
    labels = ["a"] * 10 + ["b"] * 2 + ["a"] * 10
    segs = timeline.segments(labels, np.arange(22.0), 1.0, 5.0)
    assert [s[0] for s in segs] == ["a"]                          # the 2 s run is absorbed
    segs = timeline.segments(["a"] * 10 + ["b"] * 10, np.arange(20.0), 1.0, 5.0)
    assert [(s[0], s[1], s[2]) for s in segs] == [("a", 0.0, 9.5), ("b", 9.5, 19.5)]
    summ = timeline.summary(["a"] * 10 + ["b"] * 10, segs, 1.0, 20.0)
    assert summ["time_share"] == {"a": 0.5, "b": 0.5} and summ["changes_per_minute"] == pytest.approx(3.0)


def test_smooth_labels():
    labs = ["c0", "c1", "c0", "c0", timeline.NO_FACE, "c1"]
    out = timeline.smooth_labels(labs, 3)
    assert out[1] == "c0" and out[4] == timeline.NO_FACE


def test_sampling_views():
    from common import sampling
    assert sampling.view_indices(32, 16, 0) == list(range(0, 32, 2))
    assert sampling.view_indices(32, 16, 1) == list(range(1, 32, 2))
    idx = sampling.uniform_indices(100, 32)
    assert len(idx) == 32 and idx[0] == 1 and idx[-1] == 98 and np.all(np.diff(idx) > 0)
    t = sampling.train_indices(32, 16)
    assert all(2 * i <= v <= 2 * i + 1 for i, v in enumerate(t))
