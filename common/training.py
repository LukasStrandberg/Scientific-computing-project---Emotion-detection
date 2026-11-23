"""Shared code of the two supervised tracks (S2D, MAE-DFER): data loading from the crop cache,
input transform, two-view inference and ONE fine-tuning loop.

Training protocol (both models): BCE-with-logits with pos_weight = clip(neg/pos, 1, 10) from the
training split, AdamW, linear warm-up + cosine decay (per optimiser step), AMP on CUDA, gradient
clipping, horizontal flip + temporal jitter, best epoch by validation loss (two-view averaged
logits), early stopping. Model-specific: learning rates, trainable parameters, batch size, epochs.
"""
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import cache, sampling
from common.metrics import average_precision, auc
from common.tracks import Track

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])
LOG_FIELDS = ["epoch", "train_loss", "val_loss", "val_mAP", "val_mean_auc", "lr", "minutes"]


class CacheClips(Dataset):
    """Clips from the face cache as uint8 tensors. train=True: one 16-frame view with temporal
    jitter + random horizontal flip; train=False: all 32 cached frames (both views)."""

    def __init__(self, cfg, videos, labels=None, train=False):
        self.cfg, self.videos, self.train = cfg, list(videos), train
        self.labels = None if labels is None else np.asarray(labels, np.float32)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        path = cache.clip_path(self.cfg, self.videos[i])
        if self.train:
            fr = cache.load_clip(path, sampling.train_indices(self.cfg.cache.n_frames, self.cfg.clip_len))
            if random.random() < self.cfg.train.hflip:
                fr = fr[:, :, ::-1]
        else:
            fr = cache.load_clip(path)
        y = self.labels[i] if self.labels is not None else np.zeros(6, np.float32)
        return torch.from_numpy(np.ascontiguousarray(fr)), torch.from_numpy(y), i


class ClipNet(nn.Module):
    """Wraps a video model: forward(x) -> (logits, features); probs() applies the output rule.
    output="sigmoid": 6-output multi-label head; output="softmax": source-dataset head whose
    probabilities for the 6 shared emotions (rows) are used as scores (zero-shot baselines)."""

    def __init__(self, model, output="sigmoid", rows=None):
        super().__init__()
        self.model, self.output, self.rows = model, output, rows

    def probs(self, logits):
        if self.output == "sigmoid":
            return logits.sigmoid()
        return logits.softmax(-1)[:, self.rows]


class SupervisedTrack(Track):
    supervised = True
    section = None          # name of the config section ("s2d" / "maedfer")

    @property
    def sec(self):
        return self.cfg[self.section]

    @property
    def trained_run(self):  # config <model>.run, default "finetuned"; give variants their own name
        return self.sec.run

    @property
    def runs(self):
        return ("baseline", self.trained_run)

    # ---- implemented per model
    def build(self, run):
        raise NotImplementedError

    def init_for_training(self):
        raise NotImplementedError

    def param_groups(self, net):
        raise NotImplementedError

    def set_train_mode(self, net):
        net.train()

    def cam_target(self, net, layer):
        raise NotImplementedError

    # ---- shared
    def load(self, run):
        net = self.build(run)
        return net.to(self.device).eval()

    def checkpoint(self, run):
        from common.outputs import run_dir
        return run_dir(self.cfg, self.name, run) / "best.pth"

    def to_input(self, frames):
        """uint8 (B, T, S, S, 3) on device -> normalised float (B, 3, T, s, s) at the model's input size."""
        B, T, H, W, _ = frames.shape
        x = frames.float().div_(255)
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W)
        s = self.sec.input_size
        if (H, W) != (s, s):
            x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True)
        return x.reshape(B, T, 3, s, s).permute(0, 2, 1, 3, 4).contiguous()

    def views(self, frames32):
        n = frames32.shape[1]
        return [frames32[:, sampling.view_indices(n, self.cfg.clip_len, v)] for v in range(self.cfg.eval_views)]

    def forward_views(self, net, frames32):
        """frames32 uint8 tensor (B, 32, S, S, 3) on device -> (mean logits, mean probs, mean features)."""
        amp = self.device == "cuda" and self.cfg.train.amp
        logits, probs, feats = [], [], []
        for v in self.views(frames32):
            with torch.autocast(device_type="cuda", enabled=amp):
                lg, ft = net(self.to_input(v))
            lg, ft = lg.float(), ft.float()
            logits.append(lg); probs.append(net.probs(lg)); feats.append(ft)
        return torch.stack(logits).mean(0), torch.stack(probs).mean(0), torch.stack(feats).mean(0)

    @torch.no_grad()
    def predict_frames(self, net, frames32, batch=None):
        batch = batch or self.sec.eval_batch
        P, Fs = [], []
        for i in range(0, len(frames32), batch):
            x = torch.from_numpy(np.ascontiguousarray(frames32[i:i + batch])).to(self.device)
            _, p, f = self.forward_views(net, x)
            P.append(p.cpu().numpy()); Fs.append(f.cpu().numpy())
        return {"probs": np.concatenate(P), "features": np.concatenate(Fs)}

    @torch.no_grad()
    def predict_clips(self, net, videos):
        dl = DataLoader(CacheClips(self.cfg, videos), batch_size=self.sec.eval_batch,
                        num_workers=self.cfg.train.workers, pin_memory=self.device == "cuda")
        P, Fs, t0 = [], [], time.time()
        for b, (x, _, _) in enumerate(dl, 1):
            _, p, f = self.forward_views(net, x.to(self.device, non_blocking=True))
            P.append(p.cpu().numpy()); Fs.append(f.cpu().numpy())
            if b % 25 == 0 or b == len(dl):
                print(f"  {min(b * dl.batch_size, len(videos))}/{len(videos)} clips ({time.time() - t0:.0f} s)")
        return {"probs": np.concatenate(P), "features": np.concatenate(Fs)}

    def head_from_features(self, net, feats):
        """Logits from (re-centred) clip features - used by patient.feature_recentering."""
        raise NotImplementedError

    def fit(self, split_df, out_dir, resume=False):
        return train(self, split_df, out_dir, resume)


# ------------------------------------------------------------------ the loop
def _pos_weight(cfg, Y, device):
    if not cfg.train.pos_weight:
        return None
    pos = Y.sum(0)
    return torch.tensor(np.clip((len(Y) - pos) / np.maximum(pos, 1), 1, cfg.train.pos_weight_max),
                        dtype=torch.float32, device=device)


@torch.no_grad()
def validate(track, net, loader, loss_fn):
    net.eval()
    L, Y, P, n = 0.0, [], [], 0
    for x, y, _ in loader:
        lg, p, _ = track.forward_views(net, x.to(track.device, non_blocking=True))
        y = y.to(track.device)
        L += loss_fn(lg, y).item() * len(y); n += len(y)
        Y.append(y.cpu().numpy()); P.append(p.cpu().numpy())
    Y, P = np.concatenate(Y), np.concatenate(P)
    aps = [average_precision(Y[:, i], P[:, i]) for i in range(Y.shape[1])]
    aucs = [auc(Y[:, i], P[:, i]) for i in range(Y.shape[1])]
    with np.errstate(all="ignore"):
        return L / max(n, 1), float(np.nanmean(aps)), float(np.nanmean(aucs))


def train(track, split_df, out_dir, resume=False):
    cfg, sec, dev = track.cfg, track.sec, track.device
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emo = track.emotions
    tr, va = split_df[split_df.split == "train"], split_df[split_df.split == "val"]
    Ytr = tr[emo].to_numpy(np.float32)
    w = cfg.train.workers
    train_dl = DataLoader(CacheClips(cfg, tr.video, Ytr, train=True), batch_size=sec.batch, shuffle=True,
                          drop_last=len(tr) > sec.batch, num_workers=w, pin_memory=dev == "cuda",
                          persistent_workers=w > 0)
    val_dl = DataLoader(CacheClips(cfg, va.video, va[emo].to_numpy(np.float32)), batch_size=sec.eval_batch,
                        num_workers=w, pin_memory=dev == "cuda")
    print(f"train {len(tr)} clips | val {len(va)} clips | batch {sec.batch} x accum {sec.accum}")

    net = track.init_for_training().to(dev)
    best_path = out / "best.pth"
    best, best_epoch = float("inf"), 0
    if resume and best_path.exists():
        ck = torch.load(best_path, map_location="cpu", weights_only=False)
        net.model.load_state_dict(ck["model"])
        best, best_epoch = float(ck["val_loss"]), int(ck["epoch"])
        print(f"resumed from {best_path} (epoch {best_epoch}, val loss {best:.4f}); new optimiser + schedule")

    groups = track.param_groups(net)
    trainable = [p for g in groups for p in g["params"]]
    base_lrs = [g["lr"] for g in groups]
    n_tr = sum(p.numel() for p in trainable)
    print(f"trainable parameters: {n_tr:,} of {sum(p.numel() for p in net.model.parameters()):,}")
    pw = _pos_weight(cfg, Ytr, dev)
    if pw is not None:
        print("positive-class weights:", dict(zip(emo, np.round(pw.cpu().numpy(), 2))))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay, betas=(0.9, 0.999))
    batches = len(train_dl) if not cfg.train.max_steps else min(len(train_dl), cfg.train.max_steps)
    steps_per_epoch = max(batches // sec.accum, 1)
    total, warm = sec.epochs * steps_per_epoch, cfg.train.warmup_epochs * steps_per_epoch

    def scale_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))

    amp = dev == "cuda" and cfg.train.amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log_path = out / "log.csv"
    if not (resume and log_path.exists()):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_FIELDS)

    def save(epoch, val_loss, val_map):
        torch.save({"model": net.model.state_dict(), "epoch": epoch, "val_loss": val_loss, "val_mAP": val_map,
                    "classes": emo, "model_name": track.name}, best_path)

    def log(row):
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    t0, step, since_best = time.time(), 0, 0
    if not resume:
        l0, m0, a0 = validate(track, net, val_dl, loss_fn)
        best, best_epoch = l0, 0
        save(0, l0, m0)
        log([0, "", f"{l0:.4f}", f"{m0:.4f}", f"{a0:.4f}", "", "0.0"])
        print(f"before training: val loss {l0:.4f} | val mAP {m0:.3f} | val mean AUC {a0:.3f}")
    try:
        for epoch in range(1, sec.epochs + 1):
            track.set_train_mode(net)
            run_loss, n = 0.0, 0
            opt.zero_grad(set_to_none=True)
            for b, (x, y, _) in enumerate(train_dl, 1):
                if cfg.train.max_steps and b > cfg.train.max_steps:
                    break
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=amp):
                    logits, _ = net(track.to_input(x))
                loss = loss_fn(logits.float(), y)
                scaler.scale(loss / sec.accum).backward()
                run_loss += loss.item() * len(y); n += len(y)
                if b % sec.accum == 0 or b == batches:
                    for g, lr in zip(opt.param_groups, base_lrs):
                        g["lr"] = lr * scale_at(step)
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step += 1
            vl, vm, va_auc = validate(track, net, val_dl, loss_fn)
            mins = (time.time() - t0) / 60
            lr_now = base_lrs[0] * scale_at(max(step - 1, 0))
            print(f"epoch {epoch:2d}/{sec.epochs} | train loss {run_loss / max(n, 1):.4f} | val loss {vl:.4f} | "
                  f"val mAP {vm:.3f} | val mean AUC {va_auc:.3f} | {mins:.1f} min")
            log([epoch, f"{run_loss / max(n, 1):.4f}", f"{vl:.4f}", f"{vm:.4f}", f"{va_auc:.4f}", f"{lr_now:.2e}",
                 f"{mins:.1f}"])
            if vl < best:
                best, best_epoch, since_best = vl, epoch, 0
                save(epoch, vl, vm)
                print(f"   new best (val loss) -> {best_path}")
            else:
                since_best += 1
                if cfg.train.patience and since_best >= cfg.train.patience:
                    print(f"   no improvement for {cfg.train.patience} epochs - early stop")
                    break
    except KeyboardInterrupt:
        print("interrupted - best.pth holds the best epoch so far")
    print(f"done: best epoch {best_epoch} (val loss {best:.4f}) -> {best_path}")
    return {"best_epoch": best_epoch, "best_val_loss": best, "checkpoint": str(best_path)}
