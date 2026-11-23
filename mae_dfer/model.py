"""MAE-DFER track: model wrapper, DFEW -> ElderReact head mapping, freeze modes, Grad-CAM layer.

Upstream model code is vendored unmodified in third_party/mae_dfer (MIT). Under timm 0.6.13 the
model is built by calling the registered function directly; timm.create_model would pass
`pretrained_cfg`, which the upstream VisionTransformer does not accept.
    baseline   the authors' DFEW fine-tuned model; scores = softmax shares of the 6 mapped classes
    finetuned  6-output sigmoid head initialised from the mapped DFEW rows; default freeze: all
               (head + fc_norm trained = linear probe, the MAE-DFER track's best configuration)
"""
import sys

import torch
from einops import rearrange

from common.config import REPO
from common.training import ClipNet, SupervisedTrack

THIRD = REPO / "third_party" / "mae_dfer"
if str(THIRD) not in sys.path:
    sys.path.insert(0, str(THIRD))

ARCH = dict(all_frames=16, tubelet_size=2, depth=16, attn_type="local_global",
            lg_region_size=[2, 5, 10], lg_classify_token_type="region")


def create(num_classes, drop_path=0.0):
    import modeling_finetune
    return modeling_finetune.vit_base_dim512_no_depth_patch16_160(num_classes=num_classes, drop_path_rate=drop_path,
                                                                   **ARCH)


def state_from(path, strip=("module.",)):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = next((ck[k] for k in ("model", "module", "state_dict") if isinstance(ck, dict) and k in ck), ck)
    for prefix in strip:
        sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
    return sd


def is_head(name):
    return name.startswith("head") or name.startswith("fc_norm")


class MAEDFERNet(ClipNet):
    def forward(self, x):
        feat = self.model.forward_features(x)       # (B, 512): fc_norm(mean of region tokens)
        return self.model.head(feat), feat


class MAEDFERTrack(SupervisedTrack):
    name = "maedfer"
    section = "maedfer"

    def _drop_path(self):
        return self.sec.drop_path if self.sec.freeze != "all" else 0.0

    def _dfew_model(self, drop_path=0.0):
        model = create(len(self.sec.dfew_classes), drop_path)
        path = self.cfg.resolve(self.sec.dfew_ckpt)
        if path.exists():
            msg = model.load_state_dict(state_from(path), strict=False)
            if msg.missing_keys:
                raise RuntimeError(f"{len(msg.missing_keys)} weights missing in {path} (e.g. {msg.missing_keys[:3]}) "
                                   "- architecture settings do not match this checkpoint")
            print(f"loaded {path.name} ({len(msg.unexpected_keys)} unused keys)")
        elif self.cfg.smoke_test:
            print("SMOKE_TEST: DFEW checkpoint not found - random initialisation")
        else:
            raise FileNotFoundError(f"MAE-DFER DFEW checkpoint not found: {path} (see 00_setup weights table)")
        return model

    def build(self, run):
        if run == "baseline":
            return MAEDFERNet(self._dfew_model(), "softmax", list(self.sec.dfew_rows))
        model = create(len(self.emotions))
        ck = self.checkpoint(run)
        if not ck.exists():
            raise FileNotFoundError(f"{ck} not found - run 02_train.ipynb first")
        model.load_state_dict(state_from(ck), strict=True)
        return MAEDFERNet(model, "sigmoid")

    def init_for_training(self):
        dp = self._drop_path()
        if self.sec.init == "dfew":
            state = self._dfew_model(dp).state_dict()
            rows = list(self.sec.dfew_rows)
            state["head.weight"] = state["head.weight"][rows].clone()
            state["head.bias"] = state["head.bias"][rows].clone()
            model = create(len(self.emotions), dp)
            model.load_state_dict(state, strict=True)
            return MAEDFERNet(model, "sigmoid")
        # init: pretrained - VoxCeleb2 self-supervised encoder, new head
        model = create(len(self.emotions), dp)
        path = self.cfg.resolve(self.sec.pretrained_ckpt)
        if path.exists():
            sd = state_from(path, strip=("module.", "encoder."))
            sd = {k: v for k, v in sd.items() if not k.startswith(("decoder", "head"))}
            msg = model.load_state_dict(sd, strict=False)
            bad = [k for k in msg.missing_keys if not is_head(k)]
            if bad:
                raise RuntimeError(f"{len(bad)} encoder weights not loaded from {path} (e.g. {bad[:3]})")
        elif not self.cfg.smoke_test:
            raise FileNotFoundError(path)
        return MAEDFERNet(model, "sigmoid")

    def param_groups(self, net):
        model = net.model
        n_blocks = len(model.blocks)
        head, backbone = [], []
        for name, p in model.named_parameters():
            if is_head(name):
                p.requires_grad = True
                head.append(p)
                continue
            if self.sec.freeze == "none":
                keep = True
            elif self.sec.freeze == "partial":
                keep = name.startswith("blocks.") and int(name.split(".")[1]) >= n_blocks - self.sec.unfreeze_last_blocks
            elif self.sec.freeze == "all":
                keep = False
            else:
                raise ValueError(f"unknown maedfer.freeze {self.sec.freeze!r}")
            p.requires_grad = keep
            if keep:
                backbone.append(p)
        groups = [{"params": head, "lr": self.sec.lr_head}]
        if backbone:
            groups.append({"params": backbone, "lr": self.sec.lr_backbone})
        return groups

    def set_train_mode(self, net):
        """Frozen layers stay in eval mode (no drop-path / dropout noise)."""
        net.train()
        for mod in [net.model.patch_embed] + list(net.model.blocks):
            if not any(p.requires_grad for p in mod.parameters()):
                mod.eval()

    def head_from_features(self, net, feats):
        return net.model.head(feats)

    def logit_index(self, net, c):
        return net.rows[c] if net.rows is not None else c

    def cam_target(self, net, layer="last_block"):
        """last_block: tokens entering the last LGI-Former block, regrouped from the 8 local regions
        back to the (8, 10, 10) token grid (the region token of each region is dropped).
        patch_embed: output of the 3D patch projection (Conv3d), (B, C, 8, 10, 10)."""
        m = net.model
        if layer == "last_block":
            (nt, nh, nw), (t, h, w) = m.lg_num_region_size, m.lg_region_size

            def reshape(a):   # (B * n_regions, 1 + t*h*w, C)
                return rearrange(a[:, 1:], "(b nt nh nw) (t h w) c -> b c (nt t) (nh h) (nw w)",
                                 nt=nt, nh=nh, nw=nw, t=t, h=h, w=w)
            return m.blocks[-1], "pre", reshape
        if layer == "patch_embed":
            return m.patch_embed.proj, "fwd", lambda a: a
        raise ValueError(layer)

    def checkpoint_info(self):
        return {"dfew_ckpt": str(self.cfg.resolve(self.sec.dfew_ckpt))}
