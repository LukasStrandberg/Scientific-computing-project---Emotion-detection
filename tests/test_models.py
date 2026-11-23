"""Both model definitions under the single pinned timm, built from the vendored upstream code
(random weights, CPU). Key names / shapes are compared with lists generated from the upstream
repositories at the pinned commits (tests/data/*.txt)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config import load_config

DATA = Path(__file__).parent / "data"


def _state_keys(model):
    return [(k, str(tuple(v.shape))) for k, v in model.state_dict().items()]


@pytest.fixture(scope="module")
def cfg():
    import timm
    assert timm.__version__ == "0.6.13"
    return load_config(overrides={"smoke_test": True})


def test_s2d_builds_and_matches_upstream_keys(cfg):
    from s2d.model import S2DTrack, create
    m = create(11)
    ref = [(k, s) for k, s in (line.split(" ", 1) for line in (DATA / "s2d_mafw_state_keys.txt").read_text().splitlines())]
    assert _state_keys(m) == ref
    track = S2DTrack(cfg, "cpu")
    net = track.init_for_training()                  # 6-output head from the MAFW rows
    assert net.model.head.fc_cls.out_features == 6
    x = torch.randint(0, 255, (1, 32, 224, 224, 3), dtype=torch.uint8)
    with torch.no_grad():
        logits, probs, feats = track.forward_views(net.eval(), x)
    assert logits.shape == (1, 6) and probs.shape == (1, 6) and feats.shape == (1, 768)
    n_train = sum(p.numel() for g in track.param_groups(net) for p in g["params"])
    assert 9.0e6 < n_train < 9.5e6                   # MCP + TMA + head, as in the official S2D rule


def test_maedfer_builds_and_matches_upstream_keys(cfg):
    from timm.models import create_model

    from mae_dfer.model import ARCH, MAEDFERTrack, create   # puts third_party/mae_dfer on sys.path
    import modeling_finetune  # noqa: F401,E402  (vendored; registers the model name with timm)
    m = create(7)
    ref = [tuple(line.split(" ", 1)) for line in (DATA / "maedfer_dfew_state_keys.txt").read_text().splitlines()]
    assert _state_keys(m) == ref
    assert sum(p.numel() for p in m.parameters()) == 84_878_343      # = the count printed when the DFEW ckpt loaded
    with pytest.raises(TypeError, match="pretrained_cfg"):           # the timm 0.6.13 conflict that is avoided
        create_model("vit_base_dim512_no_depth_patch16_160", num_classes=7, **ARCH)
    track = MAEDFERTrack(cfg, "cpu")
    net = track.init_for_training()
    x = torch.randint(0, 255, (2, 32, 224, 224, 3), dtype=torch.uint8)
    assert track.to_input(x[:, :16]).shape == (2, 3, 16, 160, 160)
    with torch.no_grad():
        logits, probs, feats = track.forward_views(net.eval(), x)
    assert logits.shape == (2, 6) and feats.shape == (2, 512)
    groups = track.param_groups(net)
    assert len(groups) == 1 and sum(p.numel() for p in groups[0]["params"]) == 512 * 6 + 6 + 2 * 512   # head + fc_norm


@pytest.mark.parametrize("model", ["s2d", "maedfer"])
def test_gradcam_shapes(cfg, model):
    from common import gradcam
    from common.tracks import get_track
    c = load_config(model, overrides={"smoke_test": True})
    track = get_track(c, "cpu")
    net = track.init_for_training().eval()
    frames = np.random.default_rng(0).integers(0, 255, (32, 224, 224, 3), dtype=np.uint8)
    for layer in ("last_block", "patch_embed"):
        fr16, cam, cls, probs = gradcam.gradcam(track, net, frames, layer=layer)
        assert fr16.shape == (16, 224, 224, 3) and cam.shape == (16, 224, 224)
        assert 0 <= cam.min() and cam.max() <= 1 and 0 <= cls < 6 and probs.shape == (6,)
