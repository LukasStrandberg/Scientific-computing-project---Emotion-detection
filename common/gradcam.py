"""Grad-CAM for the two video transformers (same method for S2D and MAE-DFER).

For one 16-frame view of a clip:
    1. hook the token activations A of the target layer (default: the INPUT of the last
       transformer block, see the tracks' cam_target) and keep their gradients
    2. backpropagate the logit of the target emotion (default: the highest-scoring emotion)
    3. channel weights w_c = mean of dLogit/dA_c over all tokens (space and time)
    4. CAM = ReLU(sum_c w_c A_c) on the token grid (S2D 16x14x14, MAE-DFER 8x10x10)
    5. trilinear upsampling to 16 frames x crop size, normalised to [0, 1] over the whole clip
"""
import numpy as np
import torch
import torch.nn.functional as F

from common import sampling


def gradcam(track, net, frames32, target=None, layer="last_block", view=0):
    """frames32: uint8 (32, S, S, 3). Returns (frames16 uint8, cam (16, S, S) in [0, 1], class, probs)."""
    idx = sampling.view_indices(len(frames32), track.cfg.clip_len, view)
    frames16 = np.ascontiguousarray(frames32[idx])
    S = frames16.shape[1]
    x = track.to_input(torch.from_numpy(frames16)[None].to(track.device)).requires_grad_(True)
    module, kind, reshape = track.cam_target(net, layer)
    store = {}

    def keep(t):
        t.retain_grad()
        store["a"] = t

    handle = (module.register_forward_pre_hook(lambda m, args: keep(args[0])) if kind == "pre"
              else module.register_forward_hook(lambda m, args, out: keep(out)))
    net.eval()
    try:
        with torch.enable_grad():
            net.zero_grad(set_to_none=True)
            logits, _ = net(x)
            probs = net.probs(logits.float())[0].detach().cpu().numpy()
            c = int(np.argmax(probs)) if target is None else int(target)
            logits[0, track.logit_index(net, c)].backward()
            A, G = reshape(store["a"].detach()), reshape(store["a"].grad.detach())   # (1, C, T', h, w)
            w = G.mean(dim=(2, 3, 4), keepdim=True)
            cam = torch.relu((w * A).sum(1, keepdim=True)).float()
            cam = F.interpolate(cam, size=(len(frames16), S, S), mode="trilinear", align_corners=False)[0, 0]
    finally:
        handle.remove()
        net.zero_grad(set_to_none=True)
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return frames16, cam, c, probs


def strongest_frame(cam):
    """Index of the frame with the highest mean CAM response."""
    return int(cam.reshape(len(cam), -1).mean(1).argmax())
