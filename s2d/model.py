"""S2D track: model wrapper, MAFW -> ElderReact head mapping, trainable parameters, Grad-CAM layer.

Upstream model code is vendored unmodified in third_party/s2d (Apache-2.0).
    baseline   the released MAFW model (fold 1, without oversampler); scores = softmax shares of
               the 6 emotions ElderReact shares with MAFW (zero-shot)
    finetuned  6-output sigmoid head initialised from those MAFW rows; trains the landmark prompts
               (MCP), temporal adapters (TMA) and head, ~9.3 M of 95 M parameters (official S2D rule)
"""
import sys

import torch
import torch.nn as nn

from common.config import REPO
from common.training import ClipNet, SupervisedTrack

THIRD = REPO / "third_party" / "s2d"
if str(THIRD) not in sys.path:
    sys.path.insert(0, str(THIRD))

MAFW_CLASSES = ["anger", "anxiety", "contempt", "disappointment", "disgust", "fear",
                "happiness", "helplessness", "neutral", "sadness", "surprise"]
# Same rule as the official S2D fine-tuning: parameters whose name contains one of these are trained.
TRAINABLE_KEYS = ("temporal_embedding", "temporal_attn", "cls_token_t", "ln_post", "cls_head",
                  "Adapter", "head", "prompt")
LANDMARK_WEIGHTS = THIRD / "landmark" / "mobilefacenet_model_best.pth.tar"


def create(num_classes):
    import model_finetuning  # noqa: F401  (registers s2d_base_patch16_224 with timm)
    from timm.models import create_model
    return create_model("s2d_base_patch16_224", pretrained=False, num_classes=num_classes,
                        adapter_scale=0.25, head_dropout_ratio=0.5, num_frames=16, in_chans_l=128)


def build_landmark(smoke=False):
    from landmark.backbone import Landmark
    if LANDMARK_WEIGHTS.exists():
        return Landmark("MobileFaceNet")
    if not smoke:
        raise FileNotFoundError(f"{LANDMARK_WEIGHTS} missing - run 00_setup.ipynb (downloads it)")
    from landmark.mobilefacenet import MobileFaceNet   # smoke test: same network, random weights
    lm = Landmark.__new__(Landmark)
    nn.Module.__init__(lm)
    lm.backbone, lm.size = MobileFaceNet([112, 112], 136), 112
    for p in lm.parameters():
        p.requires_grad = False
    return lm


def state_from(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("model", ck.get("model_state_dict", ck)) if isinstance(ck, dict) else ck


class S2DNet(ClipNet):
    def __init__(self, model, landmark, output="sigmoid", rows=None):
        super().__init__(model, output, rows)
        self.landmark = landmark

    def forward(self, x):
        with torch.no_grad():
            lm = self.landmark(x)
        logits, feat = self.model(x, lm)
        return logits, feat


class S2DTrack(SupervisedTrack):
    name = "s2d"
    section = "s2d"

    def _mafw_model(self):
        model = create(len(MAFW_CLASSES))
        path = self.cfg.resolve(self.sec.mafw_ckpt)
        if path.exists():
            missing, unexpected = model.load_state_dict(state_from(path), strict=False)
            if missing:
                raise RuntimeError(f"{len(missing)} weights missing when loading {path}, e.g. {missing[:3]}")
            print(f"loaded {path.name} ({len(unexpected)} unused keys)")
        elif self.cfg.smoke_test:
            print("SMOKE_TEST: MAFW checkpoint not found - random initialisation")
        else:
            raise FileNotFoundError(f"S2D MAFW checkpoint not found: {path} (see 00_setup weights table)")
        return model

    def build(self, run):
        lm = build_landmark(self.cfg.smoke_test)
        if run == "baseline":
            return S2DNet(self._mafw_model(), lm, "softmax", list(self.sec.mafw_rows))
        model = create(len(self.emotions))
        ck = self.checkpoint(run)
        if not ck.exists():
            raise FileNotFoundError(f"{ck} not found - run 02_train.ipynb first")
        model.load_state_dict(state_from(ck), strict=True)
        return S2DNet(model, lm, "sigmoid")

    def init_for_training(self):
        """6-output model; head rows copied from the MAFW classes of the 6 shared emotions."""
        state = self._mafw_model().state_dict()
        rows = list(self.sec.mafw_rows)
        state["head.fc_cls.weight"] = state["head.fc_cls.weight"][rows].clone()
        state["head.fc_cls.bias"] = state["head.fc_cls.bias"][rows].clone()
        model = create(len(self.emotions))
        model.load_state_dict(state, strict=True)
        return S2DNet(model, build_landmark(self.cfg.smoke_test), "sigmoid")

    def param_groups(self, net):
        params = []
        for name, p in net.model.named_parameters():
            p.requires_grad = (name.startswith("head.") if self.sec.trainable == "head"
                               else any(k in name for k in TRAINABLE_KEYS))
            if p.requires_grad:
                params.append(p)
        return [{"params": params, "lr": self.sec.lr}]

    def set_train_mode(self, net):
        net.train()
        net.landmark.eval()

    def head_from_features(self, net, feats):
        return net.model.head.fc_cls(feats)

    def logit_index(self, net, c):
        return net.rows[c] if net.rows is not None else c

    def cam_target(self, net, layer="last_block"):
        """(module, hook kind, reshape fn: activation -> (B, C, T, h, w)).

        last_block: the tokens entering the last ViT block. Only the CLS token is classified, so
        the patch tokens at the block OUTPUT get no gradient; at its input they still feed the CLS
        token through attention. patch_embed: output of the patch projection (low-level maps)."""
        T = self.cfg.clip_len
        if layer == "last_block":
            def reshape(a):   # (B*T, 1 + 196, C)
                bt, n, c = a.shape
                g = int(round((n - 1) ** 0.5))
                return a[:, 1:].reshape(bt // T, T, g, g, c).permute(0, 4, 1, 2, 3)
            return net.model.blocks[-1], "pre", reshape
        if layer == "patch_embed":
            def reshape(a):   # (B*T, C, 14, 14)
                bt, c, h, w = a.shape
                return a.reshape(bt // T, T, c, h, w).permute(0, 2, 1, 3, 4)
            return net.model.patch_embed.proj, "fwd", reshape
        raise ValueError(layer)

    def checkpoint_info(self):
        return {"mafw_ckpt": str(self.cfg.resolve(self.sec.mafw_ckpt))}
