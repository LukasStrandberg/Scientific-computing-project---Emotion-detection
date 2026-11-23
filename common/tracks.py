"""One small interface for the three tracks, so the notebooks call the same functions whatever
MODEL is set to.

    track = get_track(cfg)                      # cfg.model in {'s2d', 'maedfer', 'clustering'}
    track.runs                                  # runs written by 03_evaluate, e.g. ('baseline', 'finetuned')
    track.fit(split_df, out_dir, resume=False)  # 02: fine-tune / fit scaler + autoencoder + k-means
    h = track.load(run)                         # model handle for one run
    out = track.predict_clips(h, videos)        # cached ElderReact clips -> {"probs", "features"[, "cluster"]}
    out = track.predict_frames(h, frames32)     # uint8 (B, 32, S, S, 3) windows -> same
    explanation: supervised tracks -> common.gradcam.gradcam(track, h, frames32) using track.cam_target();
                 clustering -> track.profiles(h, X) (per-cluster blendshape z-scores)
"""


class Track:
    name = "base"
    supervised = True
    runs = ()
    trained_run = None          # the run produced by 02_train

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.emotions = list(cfg.emotions)

    # -- to implement
    def fit(self, split_df, out_dir, resume=False):
        raise NotImplementedError

    def load(self, run):
        raise NotImplementedError

    def predict_clips(self, handle, videos):
        raise NotImplementedError

    def predict_frames(self, handle, frames32):
        raise NotImplementedError


def get_track(cfg, device=None):
    from common.env import get_device
    device = device or get_device()
    if cfg.model == "s2d":
        from s2d.model import S2DTrack
        return S2DTrack(cfg, device)
    if cfg.model == "maedfer":
        from mae_dfer.model import MAEDFERTrack
        return MAEDFERTrack(cfg, device)
    if cfg.model == "clustering":
        from clustering.model import ClusteringTrack
        return ClusteringTrack(cfg, device)
    raise ValueError(f"unknown MODEL {cfg.model!r}")
