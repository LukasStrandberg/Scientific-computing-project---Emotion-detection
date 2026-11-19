"""Fine-tune the MAFW-trained S2D model on ElderReact (6 yes/no emotions, multi-label).

What it does
  * Starts from an S2D MAFW checkpoint (e.g. split01). The 11-class classifier is cut down
    to the 6 emotions ElderReact shares with MAFW (anger, disgust, fear, happiness,
    sadness, surprise), so training starts from what the model already knows.
  * Trains only what the official S2D code trains: the landmark prompts (MCP), the
    temporal adapters (TMA) and the classifier, ~9.3M of 95M parameters. The ViT backbone
    and the landmark detector stay frozen.
  * Loss: binary cross-entropy per emotion (option B: multi-label), with positive-class
    weights so rare emotions (fear, disgust) are not ignored.
  * Input: 16 frames spread over the whole clip ("uniform" sampling, which scored better
    on MAFW); during training each frame is picked at a random spot inside its part of the
    clip and the clip is randomly mirrored.
  * Train/validation split (--split):
      pooled   (default, same protocol as the MAE-DFER track): the official train and dev
               clips are pooled and re-split into train / validation (--val-frac, default
               15 %, fixed seed). Give --split-file to use an existing list of validation
               clips instead (e.g. the exact lists from the MAE-DFER run). The official TEST
               set is never touched.
      official the official train / dev files as given.
    The split actually used is written to the output folder (train_split.csv,
    val_split.csv, val_clips.txt), so it can be reused by others and by evaluate.py.
  * After every epoch the validation set is scored (loss and mean AUC). The best epoch
    (--select val_loss, default, or auc) is saved as best.pth, which evaluate.py --mode er
    can read. Training stops early when it has not improved for --patience epochs.

Smoke test on the 3060 (a few minutes):
    python s2d/train_elderreact.py --out results/er_ft_smoke --epochs 2 --limit-train 40 --limit-dev 40

Full run (local or Colab):
    python s2d/train_elderreact.py --out results/er_ft

Less overfitting (shorter, lower learning rate) / classifier-only comparison:
    python s2d/train_elderreact.py --out results/er_ft_short --epochs 10 --lr 5e-5 --warmup 1
    python s2d/train_elderreact.py --out results/er_ft_head --epochs 10 --lr 1e-3 --warmup 1 --trainable head

Then score it (validation first for the F1 thresholds, then test):
    python s2d/evaluate.py --mode er --sampling uniform --ckpt results/er_ft/best.pth ^
        --anno results/er_ft/val_split.csv --frames data/processed/ElderReact/faces_224 --out results/er_ft_dev
    python s2d/evaluate.py --mode er --sampling uniform --ckpt results/er_ft/best.pth ^
        --anno data/processed/ElderReact/anno/test.csv --frames data/processed/ElderReact/faces_224 ^
        --out results/er_ft_test --thresholds results/er_ft_dev/thresholds.json
"""
import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (ER_CLASSES, ER_IN_MAFW, MEAN, STD, ClipDataset, auc,  # noqa: E402
                      build_model)

# Same rule as the official S2D fine-tuning script: parameters whose name contains one of
# these are trained, everything else is frozen.
TRAINABLE_KEYS = ("temporal_embedding", "temporal_attn", "cls_token_t", "ln_post", "cls_head",
                  "Adapter", "head", "prompt")


def read_anno(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    return rows[0], rows[1:]


def write_anno(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def make_split(args, out):
    """Return (train_csv, val_csv) paths in the output folder, following --split."""
    header, train_rows = read_anno(args.train)
    _, dev_rows = read_anno(args.dev)
    if args.split == "official":
        train, val = train_rows, dev_rows
    else:
        pool = sorted(train_rows + dev_rows, key=lambda r: r[0])
        if args.split_file:
            with open(args.split_file) as f:
                # one clip per line, or CSV whose first column is the clip; extensions are ignored
                wanted = {Path(line.replace(",", " ").split()[0]).stem for line in f if line.strip()}
            wanted.discard("video_name")
            val = [r for r in pool if r[0] in wanted]
            missing = wanted - {r[0] for r in val}
            if missing:
                print(f"Warning: {len(missing)} clips in {args.split_file} are not in train+dev, e.g. {sorted(missing)[:3]}")
        else:
            names = [r[0] for r in pool]
            random.Random(args.split_seed).shuffle(names)
            val_names = set(names[:round(args.val_frac * len(names))])
            val = [r for r in pool if r[0] in val_names]
        val_set = {r[0] for r in val}
        train = [r for r in pool if r[0] not in val_set]
    train_csv, val_csv = out / "train_split.csv", out / "val_split.csv"
    write_anno(train_csv, header, train)
    write_anno(val_csv, header, val)
    (out / "val_clips.txt").write_text("\n".join(r[0] for r in val) + "\n")
    print(f"Split '{args.split}'" + (f" from {args.split_file}" if args.split_file else "")
          + f": {len(train)} train / {len(val)} validation clips (written to {out})")
    return train_csv, val_csv


class TrainClips(Dataset):
    """Uniform sampling with random jitter inside each temporal bin + random horizontal flip."""

    def __init__(self, anno_csv, frames_root, limit=0, clip_len=16, size=224):
        base = ClipDataset(anno_csv, frames_root, limit)
        self.items, self.clip_len, self.size = base.items, clip_len, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        (name, labels), d = self.items[i]
        frames = sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))
        n = len(frames)
        bin_len = n / self.clip_len
        idx = [min(int((t + random.random()) * bin_len), n - 1) for t in range(self.clip_len)]
        flip = random.random() < 0.5
        imgs = []
        for j in idx:
            img = Image.open(frames[j]).convert("RGB")
            if img.size != (self.size, self.size):
                img = img.resize((self.size, self.size), Image.BILINEAR)
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            imgs.append((np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD)
        x = np.stack(imgs).transpose(3, 0, 1, 2)  # C T H W
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(labels, dtype=torch.float32), name


def init_from_mafw(ckpt_path, device):
    """S2D with a 6-output head, initialised from an 11-class MAFW checkpoint."""
    model, landmark = build_model(ckpt_path, num_classes=11, device="cpu")
    state = model.state_dict()
    state["head.fc_cls.weight"] = state["head.fc_cls.weight"][ER_IN_MAFW].clone()
    state["head.fc_cls.bias"] = state["head.fc_cls.bias"][ER_IN_MAFW].clone()
    import model_finetuning  # noqa: F401
    from timm.models import create_model
    model6 = create_model("s2d_base_patch16_224", pretrained=False, num_classes=6,
                          adapter_scale=0.25, head_dropout_ratio=0.5, num_frames=16, in_chans_l=128)
    model6.load_state_dict(state, strict=True)
    print("Classifier initialised from MAFW classes:", ", ".join(ER_CLASSES))
    return model6.to(device), landmark.to(device).eval()


@torch.no_grad()
def score_dev(model, landmark, loader, device, loss_fn):
    """Validation: (loss, mean AUC, per-emotion AUCs). Logits are averaged over the windows."""
    model.eval()
    probs, labels, losses, n = [], [], 0.0, 0
    for x, y, _ in loader:
        B, W = x.shape[:2]
        x = x.flatten(0, 1).to(device, non_blocking=True)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            logits, _ = model(x, landmark(x))
        logits = logits.float().view(B, W, -1).mean(1)
        losses += loss_fn(logits, y.to(device)).item() * B; n += B
        probs.append(logits.sigmoid().cpu().numpy())
        labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    aucs = [auc(labels[:, i], probs[:, i]) for i in range(labels.shape[1])]
    return losses / max(n, 1), float(np.nanmean(aucs)), aucs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="external/S2D/ckpts/MAFW/without_oversampler/split01_checkpoint-best.pth")
    ap.add_argument("--train", default="data/processed/ElderReact/anno/train.csv")
    ap.add_argument("--dev", default="data/processed/ElderReact/anno/dev.csv")
    ap.add_argument("--frames", default="data/processed/ElderReact/faces_224")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2, help="clips per step (lower if CUDA out of memory)")
    ap.add_argument("--accum", type=int, default=8, help="steps per update; effective batch = batch x accum")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2, help="warm-up epochs")
    ap.add_argument("--no-pos-weight", action="store_true", help="do not up-weight rare emotions")
    ap.add_argument("--trainable", default="adapters", choices=["adapters", "head"],
                    help="adapters = MCP + TMA + classifier (official S2D); head = classifier only")
    ap.add_argument("--split", default="pooled", choices=["pooled", "official"])
    ap.add_argument("--split-file", help="pooled split: text/CSV listing the validation clips to use")
    ap.add_argument("--val-frac", type=float, default=0.15, help="pooled split: share of train+dev used for validation")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--split-only", action="store_true",
                    help="only write the train/validation split files to --out and stop (no training)")
    ap.add_argument("--select", default="val_loss", choices=["val_loss", "auc"],
                    help="which validation score picks the best epoch")
    ap.add_argument("--patience", type=int, default=8, help="stop after this many epochs without improvement (0 = off)")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-dev", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU (very slow)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_csv, val_csv = make_split(args, out)
    if args.split_only:
        return
    train_ds = TrainClips(train_csv, args.frames, args.limit_train)
    dev_ds = ClipDataset(val_csv, args.frames, args.limit_dev, n_windows=2, sampling="uniform")
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                          num_workers=args.workers, pin_memory=device == "cuda")
    dev_dl = DataLoader(dev_ds, batch_size=args.batch, num_workers=args.workers, pin_memory=device == "cuda")
    print(f"Train clips with face crops: {len(train_ds)}   validation clips: {len(dev_ds)}")

    model, landmark = init_from_mafw(args.init, device)
    for p in landmark.parameters():
        p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        p.requires_grad = (name.startswith("head.") if args.trainable == "head"
                           else any(k in name for k in TRAINABLE_KEYS))
        if p.requires_grad:
            trainable.append(p)
    n_tr = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {n_tr:,} of {sum(p.numel() for p in model.parameters()):,}")

    # positive-class weights: (#negative / #positive) per emotion, capped at 10
    y_train = np.array([lab for (_, lab), _ in train_ds.items], dtype=np.float32)
    pos = y_train.sum(0)
    pos_weight = None if args.no_pos_weight else torch.tensor(
        np.clip((len(y_train) - pos) / np.maximum(pos, 1), 1, 10), dtype=torch.float32, device=device)
    if pos_weight is not None:
        print("Positive-class weights:", dict(zip(ER_CLASSES, np.round(pos_weight.cpu().numpy(), 2))))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    steps_per_epoch = max(len(train_dl) // args.accum, 1)
    total, warm = args.epochs * steps_per_epoch, args.warmup * steps_per_epoch

    def lr_at(step):  # linear warm-up, then cosine decay
        if step < warm:
            return args.lr * (step + 1) / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))

    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_mean_auc"]
                               + [f"val_auc_{c}" for c in ER_CLASSES] + ["lr", "minutes"])

    def score_of(loss, mean_auc):  # higher = better; undefined scores never win
        v = -loss if args.select == "val_loss" else mean_auc
        return v if np.isfinite(v) else -np.inf

    step, t0, since_best = 0, time.time(), 0
    loss0, auc0, _ = score_dev(model, landmark, dev_dl, device, loss_fn)
    best = score_of(loss0, auc0)
    best_epoch = 0
    torch.save({"model": model.state_dict(), "epoch": 0, "val_loss": loss0, "val_mean_auc": auc0,
                "select": args.select, "split": args.split, "classes": ER_CLASSES, "init": args.init},
               out / "best.pth")  # starting model, replaced as soon as an epoch is better
    print(f"Before training: val loss {loss0:.4f}, val mean AUC {100 * auc0:.2f}  (best epoch chosen by {args.select})")
    for epoch in range(1, args.epochs + 1):
        model.train()
        landmark.eval()
        running, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for b, (x, y, _) in enumerate(train_dl, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.no_grad():
                lm = landmark(x)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits, _ = model(x, lm)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss / args.accum).backward()
            running += loss.item() * len(y); n += len(y)
            if b % args.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
        val_loss, val_auc, aucs = score_dev(model, landmark, dev_dl, device, loss_fn)
        mins = (time.time() - t0) / 60
        print(f"epoch {epoch:3d}/{args.epochs}  train loss {running / max(n, 1):.4f}  "
              f"val loss {val_loss:.4f}  val mean AUC {100 * val_auc:.2f}  ({mins:.1f} min)")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{running / max(n, 1):.4f}", f"{val_loss:.4f}", f"{100 * val_auc:.2f}"]
                                   + [f"{100 * a:.2f}" for a in aucs] + [f"{lr_at(step):.2e}", f"{mins:.1f}"])
        if score_of(val_loss, val_auc) > best:
            best, best_epoch, since_best = score_of(val_loss, val_auc), epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "val_mean_auc": val_auc,
                        "select": args.select, "split": args.split, "classes": ER_CLASSES, "init": args.init},
                       out / "best.pth")
            print(f"  -> new best ({args.select}), saved {out / 'best.pth'}")
        else:
            since_best += 1
            if args.patience and since_best >= args.patience:
                print(f"  no improvement for {args.patience} epochs - stopping early")
                break

    print(f"\nDone. Best epoch {best_epoch} by {args.select} "
          f"(before training: val loss {loss0:.4f}, val mean AUC {100 * auc0:.2f}). Log: {log_path}")
    if best_epoch == 0:
        print("Note: no epoch beat the starting model, so best.pth is the starting model.")


if __name__ == "__main__":
    main()
