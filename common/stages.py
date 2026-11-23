"""The notebook stages. Each notebook cell calls one of these functions; there is no logic that
exists only in a notebook. All functions are safe to re-run: finished outputs are skipped unless
redo=True.

    01_preprocess   prepare_split, build_face_cache, face_diagnostic, extract_blendshapes
    02_train        train
    03_evaluate     evaluate, evaluation_figures, metrics_table
    04_patient      patient_videos
    05_compare      compare
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import cache, data, metrics, outputs, plotting, timeline
from common.env import file_sha256, get_device, in_colab
from common.tracks import get_track


def _done(path, redo):
    if Path(path).exists() and not redo:
        print(f"already done ({path}) - skipped (redo=True to run again)")
        return True
    return False


# ================================================================== 01
def prepare_split(cfg):
    df = data.load_split(cfg)
    print(f"split file: {cfg.paths.split_file}")
    return df


def build_face_cache(cfg, df, redo=False):
    cache.load_from_drive(cfg)
    if not redo and not cache.missing_clips(cfg, df.video):
        print(f"face cache complete: {cache.cache_dir(cfg)} ({len(df)} clips)")
    else:
        from common.faces import make_detector
        det = make_detector(cfg, get_device())
        n_before = len(df) - len(cache.missing_clips(cfg, df.video))
        cache.build_cache(cfg, df, det, redo=redo)
        if redo or n_before < len(df):
            cache.save_to_drive(cfg)
    rep = pd.DataFrame(cache.read_report(cfg).values())
    rep = rep[rep["clip"].isin(set(df.video))]
    rep["frames_without_face"] = rep.frames_without_face.astype(int)
    summary = pd.Series({
        "clips": len(rep),
        "status ok": int((rep.status == "ok").sum()),
        "no face in clip (centre-crop fallback)": int((rep.status == "no_face").sum()),
        "unreadable": int((rep.status == "unreadable").sum()),
        "sampled frames without a face (filled from nearest)": f"{100 * rep.frames_without_face.mean() / cfg.cache.n_frames:.1f} %",
    }, name="value").to_frame()
    return summary


def face_diagnostic(cfg, df, n_clips=20):
    from common import diagnostics
    table, summary, (sheet, cols) = diagnostics.run(cfg, df, n_clips=n_clips, device=get_device())
    out = cfg.path("results") / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "face_detection_per_clip.csv", index=False)
    summary.to_csv(out / "face_detection_summary.csv")
    if sheet:
        plotting.finish(plotting.detection_sheet(sheet, cols), out / "face_detection_examples")
    return summary


def extract_blendshapes(cfg, df, redo=False):
    from clustering.model import extract_clip_features
    cache.ensure_cache(cfg, df.video)
    return extract_clip_features(cfg, df, redo=redo)


# ================================================================== 02
def train(cfg, resume=False, redo=False):
    track = get_track(cfg)
    df = data.load_split(cfg)
    run = track.trained_run
    out = outputs.run_dir(cfg, track.name, run)
    marker = out / ("model.pt" if not track.supervised else "log.csv")
    if resume or not _done(marker, redo):
        cache.ensure_cache(cfg, df.video)
        t0 = time.time()
        info = track.fit(df, out, resume=resume)
        print(f"training finished in {(time.time() - t0) / 60:.1f} min: {info.get('checkpoint')}")
    return training_figures(cfg, track, out)


def training_figures(cfg, track, out):
    fig_dir = out / "figures"
    if track.supervised:
        log = outputs.read_log(out)
        if log is not None and len(log) > 1:
            plotting.finish(plotting.training_curves(log), fig_dir / "training_curves")
        return log
    state = track.load()
    sw = state["k_sweep"]
    plotting.finish(plotting.k_sweep(sw["k"], sw["silhouette"], sw["inertia"], state["k"]), fig_dir / "k_sweep")
    return pd.DataFrame(sw).drop(columns="best_k")


# ================================================================== 03
def _config_extra(cfg, track, run):
    extra = {"run": run, "model": track.name, "split_file": str(cfg.paths.split_file), "cache_tag": cache.cache_tag(cfg)}
    if track.supervised:
        src = cfg.resolve(cfg.s2d.mafw_ckpt if track.name == "s2d" else cfg.maedfer.dfew_ckpt)
        if src.exists():
            extra["source_checkpoint"] = {"path": str(src), "sha256": file_sha256(src)}
    return extra


def evaluate(cfg, runs=None, redo=False):
    """Validation -> thresholds -> test for every run of the track; writes the run schema + figures."""
    track = get_track(cfg)
    df = data.load_split(cfg)
    cache.ensure_cache(cfg, df.video)
    emo = track.emotions
    for run in runs or track.runs:
        out = outputs.run_dir(cfg, track.name, run)
        if _done(out / "metrics.json", redo):
            continue
        print(f"\n=== {track.name} / {run} ===")
        handle = track.load(run)
        parts = {}
        for split in ("val", "test"):
            d = df[df.split == split]
            res = track.predict_clips(handle, list(d.video))
            parts[split] = (d, d[emo].to_numpy(np.float32), res)
        (dv, Yv, rv), (dt, Yt, rt) = parts["val"], parts["test"]
        thr = metrics.tune_thresholds(Yv, rv["probs"], emo)
        m = {"val": metrics.summarize(Yv, rv["probs"], thr, emo, n_boot=cfg.eval.n_boot, seed=cfg.eval.boot_seed),
             "test": metrics.summarize(Yt, rt["probs"], thr, emo, n_boot=cfg.eval.n_boot, seed=cfg.eval.boot_seed)}
        if not track.supervised:
            m["clustering"] = {}
            for split, (d, Y, r) in parts.items():
                ok = r["cluster"] >= 0
                m["clustering"][split] = metrics.clustering_scores(r["features"][ok], r["cluster"][ok], Y[ok])
            m["clustering"]["k"] = int(handle["k"])
        if cfg.eval.decision_rules:
            m["decision_rules"] = decision_rules(Yt, rt["probs"])
        clusters = (np.concatenate([rv["cluster"], rt["cluster"]]) if not track.supervised else None)
        pred = outputs.prediction_frame(list(dv.video) + list(dt.video), ["val"] * len(dv) + ["test"] * len(dt),
                                        np.vstack([Yv, Yt]), np.vstack([rv["probs"], rt["probs"]]), emo, clusters)
        feats = {"features": np.vstack([rv["features"], rt["features"]]).astype(np.float32),
                 "video": np.asarray(pred.video, dtype=str), "split": np.asarray(pred.split, dtype=str),
                 "labels": np.vstack([Yv, Yt]), "probs": np.vstack([rv["probs"], rt["probs"]]).astype(np.float32)}
        if clusters is not None:
            feats["cluster"] = clusters
        outputs.write_run(out, pred, m, thr, feats, {**cfg.to_dict(), **_config_extra(cfg, track, run)})
        print(f"wrote {out}")
        evaluation_figures(cfg, track, run, handle)
    if track.supervised and all((outputs.run_dir(cfg, track.name, r) / "metrics.json").exists() for r in track.runs):
        model_level_figures(cfg, track)
    return metrics_table(cfg, track.name)


def decision_rules(Y, P):
    """Isak's table: micro / macro / sample F1 for simple rules of how many emotions to output."""
    from sklearn.metrics import f1_score
    rank = np.argsort(np.argsort(-P, axis=1), axis=1)
    rules = {"thr>=0.3": (0.3, None, 0), "thr>=0.5": (0.5, None, 0), "thr>=0.7": (0.7, None, 0),
             "top-1": (0.0, 1, 0), "top-2": (0.0, 2, 0), "thr>=0.5, at least 1": (0.5, None, 1),
             "thr>=0.5, 1-2": (0.5, 2, 1)}
    out = {}
    for name, (t, max_k, min_k) in rules.items():
        pred = P >= t
        if max_k:
            pred &= rank < max_k
        if min_k:
            pred |= rank < min_k
        out[name] = {a: float(f1_score(Y, pred, average=a, zero_division=0)) for a in ("micro", "macro", "samples")}
    return out


def metrics_table(cfg, model=None, split="test"):
    rows = []
    for mdl, run in outputs.find_runs(cfg):
        if model and mdl != model:
            continue
        m = json.loads((outputs.run_dir(cfg, mdl, run) / "metrics.json").read_text())[split]
        row = {"model": mdl, "run": run}
        for k, label in metrics.METRICS:
            lo, hi = m[k]["mean_ci"]
            row[label] = f"{100 * m[k]['mean']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
        rows.append(row)
    return pd.DataFrame(rows).set_index(["model", "run"]) if rows else pd.DataFrame()


def _middle_frames(cfg, videos):
    return [cache.load_clip(cache.clip_path(cfg, v), [cfg.cache.n_frames // 2])[0] for v in videos]


def evaluation_figures(cfg, track, run, handle=None):
    """The standard figure set of one run (test split)."""
    from common import gradcam, tsne
    out = outputs.run_dir(cfg, track.name, run)
    fig_dir = out / "figures"
    emo = track.emotions
    r = outputs.read_run(out, emo)
    test = r["predictions"].split == "test"
    Y, P = r["Y"][test], r["P"][test]
    videos = list(r["predictions"].video[test])
    feats = r["features"]["features"][test]
    summ = r["metrics"]["test"]
    rng = np.random.default_rng(cfg.seed)
    plotting.finish(plotting.per_class_bars({run: summ}, emo, "auc", "ROC-AUC", chance=0.5), fig_dir / "per_class_auc")
    plotting.finish(plotting.per_class_bars({run: summ}, emo, "ap", "average precision"), fig_dir / "per_class_ap")
    pick = sorted(rng.choice(len(videos), size=min(cfg.eval.n_examples, len(videos)), replace=False))
    plotting.finish(plotting.example_clips(_middle_frames(cfg, [videos[i] for i in pick]), Y[pick], P[pick],
                                           r["thresholds"], emo, [videos[i] for i in pick]), fig_dir / "example_clips")
    ok = np.isfinite(feats).all(1)
    if ok.sum() >= 4:
        xy = np.zeros((len(feats), 2))
        xy[ok] = tsne.embed(feats[ok], cfg.seed, l2=track.supervised)
        lab = tsne.single_emotion_labels(Y, emo)
        sil = tsne.silhouette(feats[ok], np.array(lab, dtype=object)[ok], "cosine" if track.supervised else "euclidean")
        low = P.max(1) < cfg.eval.low_confidence
        panels = [dict(xy=xy[ok], labels=np.array(lab, dtype=object)[ok], kind="emotion", low_conf=low[ok],
                       note=f"silhouette (single-emotion clips) {tsne.fmt(sil)}")]
        if not track.supervised:
            cl = r["predictions"].cluster.to_numpy()[test]
            panels.insert(0, dict(xy=xy[ok], labels=cl[ok], kind="cluster"))
        plotting.finish(plotting.tsne_figure(panels, emo), fig_dir / "tsne")
    if track.supervised:
        handle = handle if handle is not None else track.load(run)
        items = []
        for i in sorted(rng.choice(len(videos), size=min(cfg.eval.gradcam_n, len(videos)), replace=False)):
            fr16, cam, c, p = gradcam.gradcam(track, handle, cache.load_clip(cache.clip_path(cfg, videos[i])))
            t = gradcam.strongest_frame(cam)
            items.append(dict(frame=fr16[t], cam=cam[t], text=f"{videos[i][:18]}\n{emo[c]} {p[c]:.2f}"))
        plotting.finish(plotting.gradcam_grid(items), fig_dir / "gradcam_grid")
    else:
        clustering_figures(cfg, track, run, r, fig_dir)


def clustering_figures(cfg, track, run, r, fig_dir):
    from clustering.blendshapes import FEATURE_NAMES
    from clustering.model import load_clip_features
    state = track.load(run)
    df = data.load_split(cfg)
    tr = df[df.split == "train"]
    X, _ = load_clip_features(cfg, list(tr.video))
    prof = track.profiles(state, X)
    plotting.finish(plotting.cluster_profiles(prof, FEATURE_NAMES, list(range(state["k"])), state["readout"],
                                              track.emotions), fig_dir / "cluster_profiles")
    sw = state["k_sweep"]
    plotting.finish(plotting.k_sweep(sw["k"], sw["silhouette"], sw["inertia"], state["k"]), fig_dir / "k_sweep")
    # representative clips: the test clips closest to each centroid
    test = r["predictions"].split == "test"
    lat, cl = r["features"]["features"][test], r["predictions"].cluster.to_numpy()[test]
    vids = np.array(r["predictions"].video[test])
    rows, n_cols = [], 5
    for k in range(state["k"]):
        m = np.where(cl == k)[0]
        if not len(m):
            rows.append((k, 0, []))
            continue
        d = ((lat[m] - state["centers"][k]) ** 2).sum(1)
        rows.append((k, len(m), _middle_frames(cfg, vids[m[np.argsort(d)[:n_cols]]])))
    plotting.finish(plotting.cluster_representatives(rows, n_cols), fig_dir / "cluster_representatives")
    log = outputs.read_log(outputs.run_dir(cfg, track.name, run))
    if log is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(plotting.SINGLE, 1.9))
        ax.plot(log.epoch, log.train_loss, color="#333333", label="train")
        ax.plot(log.epoch, log.val_loss, color="#0072B2", label="validation")
        ax.set_xlabel("epoch"); ax.set_ylabel("reconstruction MSE"); ax.legend()
        plotting.finish(fig, fig_dir / "autoencoder_loss")


def model_level_figures(cfg, track):
    """Baseline vs fine-tuned side by side (per-class bars, t-SNE)."""
    from common import tsne
    emo = track.emotions
    fig_dir = cfg.path("results") / track.name / "figures"
    runs = {r: outputs.read_run(outputs.run_dir(cfg, track.name, r), emo) for r in track.runs}
    plotting.finish(plotting.per_class_bars({r: v["metrics"]["test"] for r, v in runs.items()}, emo, "auc", "ROC-AUC",
                                            chance=0.5), fig_dir / "per_class_auc_runs")
    panels = []
    for r, v in runs.items():
        test = v["predictions"].split == "test"
        f, Y = v["features"]["features"][test], v["Y"][test]
        lab = np.array(tsne.single_emotion_labels(Y, emo), dtype=object)
        panels.append(dict(xy=tsne.embed(f, cfg.seed), labels=lab, kind="emotion",
                           low_conf=v["P"][test].max(1) < cfg.eval.low_confidence,
                           note=f"{r}: silhouette {tsne.fmt(tsne.silhouette(f, lab))}"))
    plotting.finish(plotting.tsne_figure(panels, emo), fig_dir / "tsne_runs")


# ================================================================== 04
def patient_videos(cfg, run=None, preset=None, videos=None, redo=False):
    """Timeline, segments and summary for every patient video; figures per video."""
    import tempfile
    from common.faces import make_detector
    track = get_track(cfg)
    run = run or track.trained_run
    ws = timeline.window_settings(cfg, preset)
    preset = preset or cfg.patient.preset
    vids = videos or sorted(p for p in cfg.path("patient_videos").glob(cfg.patient.video_glob)
                            if p.suffix.lower() in data.VIDEO_EXTS)
    if not vids:
        raise FileNotFoundError(f"no videos in {cfg.path('patient_videos')}")
    out = outputs.run_dir(cfg, track.name, run) / "patients" / preset
    print(f"{len(vids)} videos | windows {ws['window']:g} s every {ws['stride']:.3f} s, smoothing {ws['smooth']} "
          f"| run {track.name}/{run} -> {out}")
    handle, detector, results = None, None, {}
    for vid in map(Path, vids):
        if _done(out / f"{vid.stem}_summary.json", redo):   # written last, after the figures
            results[vid.stem] = load_timeline(cfg, out, vid.stem, track.supervised, ws)
            continue
        handle = handle if handle is not None else track.load(run)
        detector = detector or make_detector(cfg, get_device())
        tmp_dir = Path(cfg.paths.colab_local_cache) if in_colab() else cfg.path("cache")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=tmp_dir)
        tmp.close()
        try:
            t0 = time.time()
            print(f"{vid.name}: face crops every {ws['step']:.3f} s ...")
            grid = timeline.crop_video_grid(cfg, vid, ws["step"], detector, tmp.name)
            win = timeline.window_table(grid, ws, cfg.patient.min_face)
            print(f"  {len(grid['crops'])} frames ({100 * grid['has_face'].mean():.1f} % with a face), "
                  f"{len(win['starts'])} windows ({time.time() - t0:.0f} s)")
            if not win["starts"]:
                print("  shorter than one window - skipped")
                continue
            res = score_windows(cfg, track, handle, grid, win, ws)
            tl = write_timeline(cfg, track, out, vid.stem, win, res, ws, grid["duration"])
            results[vid.stem] = tl
            patient_figures(cfg, track, handle, vid, grid, win, res, tl, out / "figures")
            (out / f"{vid.stem}_summary.json").write_text(json.dumps(tl["summary"], indent=2))
            print(f"  done in {(time.time() - t0) / 60:.1f} min")
        finally:
            grid = None
            import gc
            gc.collect()
            Path(tmp.name).unlink(missing_ok=True)
    return pd.DataFrame({v: {k: tl["summary"][k] for k in ("windows", "segments", "changes_per_minute", "mean_segment_s")}
                         for v, tl in results.items()}).T


def score_windows(cfg, track, handle, grid, win, ws):
    valid, starts, n = win["valid"], win["starts"], ws["n"]
    W = len(starts)
    if track.supervised:
        probs, feats = np.full((W, 6), np.nan, np.float32), None
        for b, frames in timeline.iter_windows(grid["crops"], starts, n, batch=track.sec.eval_batch):
            r = track.predict_frames(handle, frames)
            feats = feats if feats is not None else np.full((W, r["features"].shape[1]), np.nan, np.float32)
            probs[b:b + len(frames)], feats[b:b + len(frames)] = r["probs"], r["features"]
        if cfg.patient.feature_recentering:
            probs = recenter(cfg, track, handle, feats, valid)
        probs[~valid] = np.nan
        return {"probs": probs, "features": feats}
    from clustering import blendshapes as bs
    ext = bs.make_extractor(cfg)
    pf = np.full((len(grid["crops"]), len(bs.BLENDSHAPES)), np.nan, np.float32)
    for i in range(0, len(pf), 64):
        pf[i:i + 64] = ext(np.asarray(grid["crops"][i:i + 64]))
    pf[~grid["has_face"]] = np.nan
    X = np.stack([bs.aggregate(pf[s:s + n]) for s in starts])
    r = track.predict_features(handle, X)
    r["probs"][~valid] = np.nan
    r["cluster"][~valid] = -1
    return r


def recenter(cfg, track, net, feats, valid):
    """Isak's test-time adaptation: standardise this video's features and map them onto the mean / std
    of the ElderReact features of the same run (val + test clips), then apply the head again."""
    import torch
    src = outputs.read_run(outputs.run_dir(cfg, track.name, track.trained_run), track.emotions)["features"]["features"]
    mu_s, sd_s = src.mean(0), src.std(0)
    f = feats[valid]
    z = (f - f.mean(0)) / np.maximum(f.std(0), 1e-6) * sd_s + mu_s
    probs = np.full((len(feats), 6), np.nan, np.float32)
    with torch.no_grad():
        lg = track.head_from_features(net, torch.tensor(z, dtype=torch.float32, device=track.device))
        probs[valid] = net.probs(lg.float()).cpu().numpy()
    return probs


def write_timeline(cfg, track, out, stem, win, res, ws, duration):
    emo = track.emotions
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"start_s": np.round(win["start_s"], 3), "end_s": np.round(win["end_s"], 3),
                       "centre_s": np.round(win["centre_s"], 3), "face_share": np.round(win["face_share"], 3)})
    for i, e in enumerate(emo):
        df[f"prob_{e}"] = np.round(res["probs"][:, i], 5)
    if track.supervised:
        sm = timeline.smooth(res["probs"], ws["smooth"])
        labels = timeline.dominant(sm, emo, cfg.patient.min_score)
    else:
        df["cluster"] = res["cluster"]
        raw = [timeline.NO_FACE if c < 0 else f"cluster {c}" for c in res["cluster"]]
        labels = timeline.smooth_labels(raw, ws["smooth"])
        sm = None
    df["predicted"] = labels
    df.to_csv(out / f"{stem}_timeline.csv", index=False)
    segs = timeline.segments(labels, win["centre_s"], ws["stride"], cfg.patient.min_segment)
    pd.DataFrame(segs, columns=["label", "start_s", "end_s"]).round(2).to_csv(out / f"{stem}_segments.csv", index=False)
    summ = timeline.summary(labels, segs, ws["stride"], duration)
    return _tl_dict(win["centre_s"], sm, df, segs, ws, duration, summ, win["valid"])


def _tl_dict(centre, sm, df, segs, ws, duration, summ, valid):
    return {"centre_s": np.asarray(centre), "probs_smooth": sm,
            "clusters": df["cluster"].replace(-1, np.nan).to_numpy(float) if "cluster" in df else None,
            "segments": segs, "stride": ws["stride"], "duration": duration, "summary": summ, "valid": np.asarray(valid)}


def load_timeline(cfg, out, stem, supervised, ws):
    df = pd.read_csv(out / f"{stem}_timeline.csv")
    segs = [tuple(r) for r in pd.read_csv(out / f"{stem}_segments.csv").itertuples(index=False)]
    summ = json.loads((out / f"{stem}_summary.json").read_text())
    P = df[[f"prob_{e}" for e in cfg.emotions]].to_numpy(float)
    sm = timeline.smooth(P, ws["smooth"]) if supervised else None
    valid = df.face_share.to_numpy() >= cfg.patient.min_face
    return _tl_dict(df.centre_s, sm, df, segs, ws, summ["duration_s"], summ, valid)


def patient_figures(cfg, track, handle, vid, grid, win, res, tl, fig_dir):
    from common import gradcam, sampling, tsne
    emo, stem = track.emotions, vid.stem
    kind = "emotion" if track.supervised else "cluster"
    fig = plotting.timeline_figure(tl, emo, kind, cfg.patient.min_score)
    plotting.finish(fig, fig_dir / f"{stem}_timeline")
    # annotated frames at the configured moments (window closest to each moment)
    cen, rows = win["centre_s"], []
    idx_frames = {}
    for m in cfg.patient.moments:
        w = int(np.argmin(np.abs(cen - m * tl["duration"])))
        g = win["starts"][w] + cfg.cache.n_frames // 2
        idx_frames[w] = (int(round(grid["times"][g] * grid["fps"])), g)
    full = sampling.read_frames_at(vid, [f for f, _ in idx_frames.values()])
    for w, (fi, g) in idx_frames.items():
        if fi in full:
            rows.append(dict(image=full[fi][..., ::-1], box=grid["boxes"][g], probs=res["probs"][w], time_s=grid["times"][g]))
    if rows:
        plotting.finish(plotting.annotated_frames(rows, emo), fig_dir / f"{stem}_annotated_frames")
    # review sheets: a stratified sample of windows (by predicted label)
    labels = np.array(pd.read_csv(fig_dir.parent / f"{stem}_timeline.csv").predicted)
    rng = np.random.default_rng(cfg.seed)
    chosen = []
    for lab in sorted(set(labels) - {timeline.NO_FACE}):
        mem = np.where(labels == lab)[0]
        k = min(len(mem), max(cfg.patient.review_min_per_class, int(round(cfg.patient.review_fraction * len(mem)))))
        chosen += rng.choice(mem, size=k, replace=False).tolist()
    for w in sorted(chosen):
        s = win["starts"][w]
        n = cfg.cache.n_frames
        fr16 = np.asarray(grid["crops"][s:s + n])[sampling.view_indices(n, cfg.clip_len, 0)]
        info = f"{stem} | window {w} | {timeline.fmt_time(win['start_s'][w])}-{timeline.fmt_time(win['end_s'][w])} | {labels[w]}"
        plotting.finish(plotting.review_sheet(fr16, res["probs"][w], emo, info), fig_dir / "review" / f"{stem}_{w:05d}",
                        show=False)
    print(f"  {len(chosen)} review sheets in {fig_dir / 'review'}")
    valid = np.where(win["valid"])[0]
    if track.supervised and len(valid):
        items = []
        for w in sorted(rng.choice(valid, size=min(cfg.patient.gradcam_n, len(valid)), replace=False)):
            s = win["starts"][w]
            fr16, cam, c, p = gradcam.gradcam(track, handle, np.asarray(grid["crops"][s:s + cfg.cache.n_frames]))
            t = gradcam.strongest_frame(cam)
            items.append(dict(frame=fr16[t], cam=cam[t], text=f"{timeline.fmt_time(win['start_s'][w])}  {emo[c]} {p[c]:.2f}"))
        plotting.finish(plotting.gradcam_grid(items), fig_dir / f"{stem}_gradcam_grid")
    feats = res["features"]
    ok = np.isfinite(feats).all(1) if feats is not None else np.zeros(0, bool)
    if ok.sum() >= 4:
        xy = tsne.embed(feats[ok], cfg.seed, l2=track.supervised)
        if track.supervised:
            P = res["probs"][ok]
            panel = dict(xy=xy, labels=np.array([emo[i] for i in P.argmax(1)], dtype=object), kind="emotion",
                         low_conf=P.max(1) < cfg.eval.low_confidence)
        else:
            panel = dict(xy=xy, labels=res["cluster"][ok], kind="cluster")
        plotting.finish(plotting.tsne_figure([panel], emo), fig_dir / f"{stem}_tsne")


# ================================================================== 05
def compare(cfg, runs=None, reference=None, preset=None):
    """Cross-model comparison from the run schema only. runs: list of 'model/run' (default: all)."""
    from common import tsne
    emo = list(cfg.emotions)
    found = [f"{m}/{r}" for m, r in outputs.find_runs(cfg)]
    runs = runs or found
    missing = [r for r in runs if r not in found]
    if missing:
        raise FileNotFoundError(f"runs without results: {missing} (available: {found})")
    out = cfg.path("results") / "compare"
    loaded = {r: outputs.read_run(cfg.path("results") / r, emo) for r in runs}
    tests = {}
    for name, v in loaded.items():
        df = v["predictions"]
        t = (df.split == "test").to_numpy()
        tests[name] = (df.video[t].tolist(), v["Y"][t], v["P"][t], v["thresholds"])
    ref = reference or runs[0]
    order, Y_ref = tests[ref][0], tests[ref][1]
    pos_ref = {v: i for i, v in enumerate(order)}
    Y_ref = Y_ref[[pos_ref[v] for v in order]]
    aligned = {}
    for name, (vids, Y, P, thr) in tests.items():
        if set(vids) != set(order):
            raise ValueError(f"{name} was not evaluated on the same test clips as {ref}")
        pos = {v: i for i, v in enumerate(vids)}
        ii = [pos[v] for v in order]
        aligned[name] = (Y[ii], P[ii], thr)
        if not np.array_equal(Y[ii], Y_ref):
            raise ValueError(f"{name}: test labels differ from {ref}")
    # tables
    table = metrics_table(cfg)
    table = table.loc[[tuple(r.split("/")) for r in runs]]
    names = list(aligned)
    rows = metrics.paired_differences([aligned[n] for n in names], emo, ref=names.index(ref),
                                      n_boot=cfg.eval.n_boot, seed=cfg.eval.boot_seed)
    for r in rows:
        r["run"] = names[r.pop("run_index")]
    diffs = pd.DataFrame(rows)[["run", "label", "diff", "lo", "hi", "significant"]]
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "test_metrics.csv")
    diffs.to_csv(out / f"paired_differences_vs_{ref.replace('/', '_')}.csv", index=False)
    per_class = pd.DataFrame({n: {e: loaded[n]["metrics"]["test"]["auc"]["per_class"][e] for e in emo} for n in names})
    per_class.to_csv(out / "test_auc_per_class.csv")
    # figures
    fig_dir = out / "figures"
    summ = {n: loaded[n]["metrics"]["test"] for n in names}
    plotting.finish(plotting.per_class_bars(summ, emo, "auc", "ROC-AUC", chance=0.5), fig_dir / "per_class_auc")
    plotting.finish(plotting.per_class_bars(summ, emo, "ap", "average precision"), fig_dir / "per_class_ap")
    plotting.finish(plotting.diff_forest([r for r in rows if r["label"] in ("ROC-AUC", "AP")], ref), fig_dir / "paired_differences")
    panels = []
    for n in names:
        v = loaded[n]
        t = (v["predictions"].split == "test").to_numpy()
        f = v["features"]["features"][t]
        ok = np.isfinite(f).all(1)
        lab = np.array(tsne.single_emotion_labels(v["Y"][t], emo), dtype=object)[ok]
        panels.append(dict(xy=tsne.embed(f[ok], cfg.seed, l2=not n.startswith("clustering")), labels=lab, kind="emotion",
                           note=n))
    plotting.finish(plotting.tsne_figure(panels, emo), fig_dir / "tsne")
    compare_patients(cfg, names, preset, fig_dir)
    return table, diffs


def compare_patients(cfg, names, preset, fig_dir):
    preset = preset or cfg.patient.preset
    ws = timeline.window_settings(cfg, preset)
    stems = set()
    for n in names:
        stems |= {p.name[:-len("_timeline.csv")] for p in (cfg.path("results") / n / "patients" / preset).glob("*_timeline.csv")}
    for stem in sorted(stems):
        items, shares = [], {}
        for n in names:
            d = cfg.path("results") / n / "patients" / preset
            if not (d / f"{stem}_timeline.csv").exists():
                continue
            supervised = not n.startswith("clustering/")
            tl = load_timeline(cfg, d, stem, supervised, ws)
            items.append((n, tl, "emotion" if supervised else "cluster"))
            shares[n] = tl["summary"]["time_share"]
        if items:
            plotting.finish(plotting.stacked_timelines(items, list(cfg.emotions), cfg.patient.min_score),
                            fig_dir / f"{stem}_timelines")
            plotting.finish(plotting.time_share_bars(shares), fig_dir / f"{stem}_time_share")
