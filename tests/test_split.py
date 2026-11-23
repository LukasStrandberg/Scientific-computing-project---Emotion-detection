import random

import pandas as pd

from common.config import REPO
from common.data import make_pooled_split, read_labels, stem

SHIPPED = REPO / "splits" / "elderreact_pooled_seed42.csv"


def _rows(prefix, n):
    return [(f"{prefix}_{i}", [i % 2, 0, 0, 1, 0, 0]) for i in range(n)]


def test_split_is_deterministic_and_sized():
    tr, dv, te = _rows("tr", 615), _rows("dv", 355), _rows("te", 353)
    a = make_pooled_split(tr, dv, te, 0.15, 42)
    b = make_pooled_split(tr, dv, te, 0.15, 42)
    assert a == b
    counts = pd.Series(a).value_counts()
    assert counts["val"] == round(0.15 * 970) == 146
    assert counts["train"] == 824 and counts["test"] == 353
    assert all(a[c] == "test" for c, _ in te)            # official test untouched
    assert make_pooled_split(tr, dv, te, 0.15, 7) != a    # the seed matters


def test_split_rule_matches_mae_dfer_track():
    """Pool in label-file order, random.Random(seed).shuffle, first n = validation."""
    tr, dv, te = _rows("tr", 20), _rows("dv", 10), _rows("te", 5)
    pool = [c for c, _ in tr + dv]
    random.Random(42).shuffle(pool)
    split = make_pooled_split(tr, dv, te, 0.15, 42)
    assert {c for c, s in split.items() if s == "val"} == set(pool[:round(0.15 * 30)])


def test_shipped_split_reproduces_previous_validation_set():
    df = pd.read_csv(SHIPPED)
    assert df.split.value_counts().to_dict() == {"train": 824, "test": 353, "val": 146}
    # first validation clips printed by the MAE-DFER notebook (Isaks_branch, MAE-DFER.ipynb)
    assert df.video.tolist()[:5] == ["heartbeat_23", "Branda_schmitz_36", "would_you_rather_36",
                                     "dolan_dark_60", "godofwar_50"]
    assert df.video.is_unique


def test_label_parsing(tmp_path):
    p = tmp_path / "train_labels.txt"
    p.write_text("50_50_100.mp4 0 1 1 0 0 1 f 1.66\n\n50_50_104.mp4 1 0 0 0 0 0 m 2.33\n")
    assert read_labels(p) == [("50_50_100", [0, 1, 1, 0, 0, 1]), ("50_50_104", [1, 0, 0, 0, 0, 0])]
    assert stem("a/b/clip_1.mp4") == "clip_1" and stem("clip.v2") == "clip.v2"
