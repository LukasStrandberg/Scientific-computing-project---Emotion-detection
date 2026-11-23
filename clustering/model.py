"""Clustering track: blendshape clip vectors -> StandardScaler -> autoencoder (or PCA) -> k-means.

fit (02_train), on the TRAINING clips only:
    1. 156-d [mean, max, std] blendshape vector per clip (clips without any detected face are left out)
    2. StandardScaler; autoencoder 156 -> 64 -> 32 -> latent (8) -> 32 -> 64 -> 156 (MSE, Adam), or PCA
    3. k-means on the latent vectors; k = best silhouette over k_range (or the configured k)
    4. read-out P(emotion | cluster) = (positives + a) / (members + 2a) on the training clips; this turns
       a cluster into 6 scores, so the same AUC / mAP / F1 (with CIs) as for the supervised tracks apply
predict: cluster = nearest centroid; probs = read-out of that cluster (clips without a face: base rates).
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from clustering import blendshapes as bs
from common import cache, sampling
from common.tracks import Track


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden=(64, 32), latent_dim=8):
        super().__init__()
        dims = [input_dim, *hidden]
        enc, dec = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        self.encoder = nn.Sequential(*enc, nn.Linear(dims[-1], latent_dim))
        rdims = [latent_dim, *reversed(hidden)]
        for a, b in zip(rdims[:-1], rdims[1:]):
            dec += [nn.Linear(a, b), nn.ReLU()]
        self.decoder = nn.Sequential(*dec, nn.Linear(rdims[-1], input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def features_path(cfg):
    return cfg.path("cache") / "blendshapes" / f"{cache.cache_tag(cfg)}_{cfg.clustering.source}.npz"


def extract_clip_features(cfg, df, redo=False):
    """Per-frame blendshapes for all clips of df (from the shared crop cache). Saved once."""
    out = features_path(cfg)
    if out.exists() and not redo:
        d = np.load(out)
        if set(df.video) <= set(d["videos"].tolist()):
            print(f"blendshape features already extracted ({out})")
            return out
    ext = bs.make_extractor(cfg)
    videos, per_frame, t0 = list(df.video), [], time.time()
    paths = dict(zip(df.video, df.path))
    for n, v in enumerate(videos, 1):
        if cfg.clustering.source == "full_frame":
            meta = cache.load_meta(cache.clip_path(cfg, v))
            frames = sampling.read_frames_at(paths[v], meta["frame_idx"])
            imgs = np.stack([frames[int(i)][..., ::-1] for i in meta["frame_idx"] if int(i) in frames])
        else:
            imgs = cache.load_clip(cache.clip_path(cfg, v))
        pf = ext(imgs)
        if len(pf) < cfg.cache.n_frames:
            pf = np.vstack([pf, np.full((cfg.cache.n_frames - len(pf), pf.shape[1]), np.nan, np.float32)])
        per_frame.append(pf)
        if n % 100 == 0 or n == len(videos):
            print(f"  [{n}/{len(videos)}] blendshapes ({time.time() - t0:.0f} s)")
    per_frame = np.stack(per_frame)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, videos=np.array(videos), per_frame=per_frame)
    found = np.isfinite(per_frame).all(2)
    print(f"saved {out}: face found by MediaPipe on {100 * found.mean():.1f} % of frames, "
          f"{int((~found.any(1)).sum())} clips without any")
    return out


def load_clip_features(cfg, videos):
    d = np.load(features_path(cfg))
    pos = {v: i for i, v in enumerate(d["videos"].tolist())}
    missing = [v for v in videos if v not in pos]
    if missing:
        raise FileNotFoundError(f"no blendshape features for {len(missing)} clips (e.g. {missing[:3]}) - "
                                "run 01_preprocess.ipynb with MODEL = 'clustering'")
    pf = d["per_frame"][[pos[v] for v in videos]]
    return np.stack([bs.aggregate(p) for p in pf]), pf


class ClusteringTrack(Track):
    name = "clustering"
    supervised = False

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.c = cfg.clustering
        self.runs = (self.c.run,)
        self.trained_run = self.c.run

    def model_path(self, run=None):
        from common.outputs import run_dir
        return run_dir(self.cfg, self.name, run or self.c.run) / "model.pt"

    # ------------------------------------------------------------ fit
    def fit(self, split_df, out_dir, resume=False):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tr, va = split_df[split_df.split == "train"], split_df[split_df.split == "val"]
        Xtr, _ = load_clip_features(self.cfg, list(tr.video))
        Xva, _ = load_clip_features(self.cfg, list(va.video))
        Ytr = tr[self.emotions].to_numpy(np.float32)
        ok, okv = np.isfinite(Xtr).all(1), np.isfinite(Xva).all(1)
        print(f"fit on {ok.sum()} training clips with a face ({(~ok).sum()} without are left out)")
        mean, scale = Xtr[ok].mean(0), Xtr[ok].std(0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        Ztr, Zva = (Xtr[ok] - mean) / scale, (Xva[okv] - mean) / scale
        state = {"mean": mean, "scale": scale, "embedding": self.c.embedding, "feature_names": bs.FEATURE_NAMES}
        if self.c.embedding == "ae":
            ae, log = self._train_ae(Ztr, Zva)
            state["ae"] = {k: v.cpu() for k, v in ae.state_dict().items()}
            with open(out / "log.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "val_loss"])
                w.writerows(log)
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.c.latent_dim, random_state=self.cfg.seed).fit(Ztr)
            state["pca"] = {"components": pca.components_, "mean": pca.mean_}
        self.state = state
        L = self._latent(Ztr)
        sweep = self._k_sweep(L)
        k = sweep["best_k"] if str(self.c.k) == "auto" else int(self.c.k)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=self.cfg.seed, n_init=10).fit(L)
        a = float(self.c.readout_prior)
        y = Ytr[ok]
        readout = np.stack([(y[km.labels_ == c].sum(0) + a) / ((km.labels_ == c).sum() + 2 * a) for c in range(k)])
        state.update(centers=km.cluster_centers_, k=k, k_sweep=sweep, readout=readout, base_rate=Ytr.mean(0))
        torch.save(state, out / "model.pt")
        print(f"k = {k} ({'best silhouette' if str(self.c.k) == 'auto' else 'configured'}); saved {out / 'model.pt'}")
        return {"k": k, "k_sweep": sweep, "checkpoint": str(out / "model.pt")}

    def _train_ae(self, Ztr, Zva):
        torch.manual_seed(self.cfg.seed)
        c = self.c
        ae = Autoencoder(Ztr.shape[1], tuple(c.hidden), c.latent_dim)
        opt = torch.optim.Adam(ae.parameters(), lr=c.ae_lr, weight_decay=1e-5)
        X, Xv = torch.tensor(Ztr, dtype=torch.float32), torch.tensor(Zva, dtype=torch.float32)
        g = torch.Generator().manual_seed(self.cfg.seed)
        log = []
        for ep in range(1, c.ae_epochs + 1):
            ae.train()
            perm, tot = torch.randperm(len(X), generator=g), 0.0
            for i in range(0, len(X), c.ae_batch):
                xb = X[perm[i:i + c.ae_batch]]
                loss = nn.functional.mse_loss(ae(xb), xb)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(xb)
            if ep % 10 == 0 or ep == c.ae_epochs or ep == 1:
                ae.eval()
                with torch.no_grad():
                    vl = nn.functional.mse_loss(ae(Xv), Xv).item() if len(Xv) else float("nan")
                log.append([ep, f"{tot / len(X):.5f}", f"{vl:.5f}"])
        print(f"autoencoder: final reconstruction loss train {log[-1][1]} / val {log[-1][2]}")
        return ae.eval(), log

    def _k_sweep(self, L):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        lo, hi = self.c.k_range
        ks, sils, inertias = [], [], []
        for k in range(lo, min(hi, len(L) - 1) + 1):
            km = KMeans(n_clusters=k, random_state=self.cfg.seed, n_init=10).fit(L)
            ks.append(k); inertias.append(float(km.inertia_))
            sils.append(float(silhouette_score(L, km.labels_)) if len(set(km.labels_)) > 1 else float("nan"))
        best = ks[int(np.nanargmax(sils))] if ks else 2
        return {"k": ks, "silhouette": sils, "inertia": inertias, "best_k": best}

    # ------------------------------------------------------------ inference
    def _latent(self, Z):
        st = self.state
        if st["embedding"] == "ae":
            c = self.c
            ae = Autoencoder(Z.shape[1], tuple(c.hidden), c.latent_dim)
            ae.load_state_dict(st["ae"])
            with torch.no_grad():
                return ae.encoder(torch.tensor(Z, dtype=torch.float32)).numpy()
        return (Z - st["pca"]["mean"]) @ st["pca"]["components"].T

    def load(self, run=None):
        p = self.model_path(run)
        if not p.exists():
            raise FileNotFoundError(f"{p} not found - run 02_train.ipynb with MODEL = 'clustering'")
        self.state = torch.load(p, map_location="cpu", weights_only=False)
        return self.state

    def predict_features(self, state, X):
        """X (N, 156) -> dict(probs (N,6), features = latent (N,d), cluster (N,), -1 / base rates if no face)."""
        self.state = state
        ok = np.isfinite(X).all(1)
        d = self.c.latent_dim
        latent = np.full((len(X), d), np.nan, np.float32)
        cluster = np.full(len(X), -1, int)
        probs = np.tile(state["base_rate"], (len(X), 1)).astype(np.float32)
        if ok.any():
            L = self._latent((X[ok] - state["mean"]) / state["scale"])
            dist = ((L[:, None, :] - state["centers"][None]) ** 2).sum(-1)
            cl = dist.argmin(1)
            latent[ok], cluster[ok], probs[ok] = L, cl, state["readout"][cl]
        return {"probs": probs, "features": latent, "cluster": cluster}

    def predict_clips(self, state, videos):
        X, _ = load_clip_features(self.cfg, list(videos))
        return self.predict_features(state, X)

    def predict_frames(self, state, frames32):
        ext = bs.make_extractor(self.cfg)
        X = np.stack([bs.aggregate(ext(f)) for f in frames32])
        return self.predict_features(state, X)

    def profiles(self, state, X):
        """Mean z-score of each blendshape feature per cluster (training clips) -> (k, 156)."""
        out = self.predict_features(state, X)
        Z = (X - state["mean"]) / state["scale"]
        return np.stack([np.nanmean(Z[out["cluster"] == c], 0) if (out["cluster"] == c).any()
                         else np.zeros(Z.shape[1]) for c in range(state["k"])])
