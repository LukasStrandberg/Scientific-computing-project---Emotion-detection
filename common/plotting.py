"""One plotting style and every shared figure.

Style: common/emotion.mplstyle (serif, 8-9 pt, two-column LaTeX widths SINGLE = 3.4 in and
FULL = 7.0 in). Every emotion has one fixed colour-blind-safe colour (Okabe-Ito) and marker in all
figures and models. Clusters use a separate palette (Paul Tol "muted") so they are never mistaken
for emotions. Perceptually uniform colormaps only (cividis for Grad-CAM, viridis for sequential
data), always with a colourbar. No in-figure titles unless config plot.titles is true; multi-panel
figures get (a), (b), ... labels. finish() saves PDF + 300 dpi PNG.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

STYLE_FILE = Path(__file__).with_name("emotion.mplstyle")
SINGLE, FULL = 3.4, 7.0
EMOTION_COLORS = {"anger": "#D55E00", "disgust": "#009E73", "fear": "#CC79A7", "happiness": "#E69F00",
                  "sadness": "#0072B2", "surprise": "#56B4E9"}
EMOTION_MARKERS = {"anger": "o", "disgust": "s", "fear": "^", "happiness": "D", "sadness": "v", "surprise": "P"}
OTHER, NO_FACE_COLOR, TEXT_MUTED = "#BBBBBB", "#EEEEEE", "#555555"
CLUSTER_COLORS = ["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE", "#882255", "#44AA99", "#999933", "#AA4499"]
CLUSTER_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p"]
CAM_CMAP = "cividis"
_OPTS = {"titles": False, "formats": ["pdf", "png"], "dpi": 300}


def use_style(cfg=None):
    plt.style.use(str(STYLE_FILE))
    if cfg is not None:
        _OPTS.update(titles=bool(cfg.plot.titles), formats=list(cfg.plot.formats), dpi=int(cfg.plot.dpi))


def cluster_color(k):
    return CLUSTER_COLORS[int(k) % len(CLUSTER_COLORS)]


def cluster_marker(k):
    return CLUSTER_MARKERS[int(k) % len(CLUSTER_MARKERS)]


def label_color(lab):
    if lab in EMOTION_COLORS:
        return EMOTION_COLORS[lab]
    if isinstance(lab, str) and lab.startswith("cluster "):
        return cluster_color(int(lab.split()[1]))
    return NO_FACE_COLOR if lab == "no face" else OTHER


def title(ax, text, **kw):
    if _OPTS["titles"] and text:
        ax.set_title(text, **kw)


def panel_labels(axes, x=-0.02, y=1.02):
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(x, y, f"({chr(97 + i)})", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)


def finish(fig, path, show=True):
    """Save as PDF + PNG (path without extension) and show inline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in _OPTS["formats"]:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=_OPTS["dpi"])
    if show and "agg" not in matplotlib.get_backend().lower():
        plt.show()
    plt.close(fig)
    return path.with_suffix(".png")


def emotion_legend(ax_or_fig, emotions, extra=(), **kw):
    h = [Line2D([], [], ls="", marker=EMOTION_MARKERS[e], color=EMOTION_COLORS[e], label=e) for e in emotions]
    h += list(extra)
    return ax_or_fig.legend(handles=h, **kw)


# ------------------------------------------------------------------ training / metrics
def training_curves(log):
    """log: DataFrame from log.csv. (a) train/val loss, best epoch marked; (b) validation mAP and AUC."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.1))
    tr = log[log.train_loss.astype(str) != ""].copy()
    tr = tr[tr.epoch > 0]
    a.plot(tr.epoch, tr.train_loss.astype(float), color="#333333", label="train")
    a.plot(log.epoch, log.val_loss.astype(float), color="#0072B2", label="validation")
    best = log.loc[log.val_loss.astype(float).idxmin()]
    a.plot([best.epoch], [float(best.val_loss)], ls="", marker="o", mfc="none", color="#0072B2", ms=6,
           label=f"best (epoch {int(best.epoch)})")
    a.set_xlabel("epoch"); a.set_ylabel("BCE loss"); a.legend()
    b.plot(log.epoch, log.val_mAP.astype(float), color="#009E73", label="mAP")
    b.plot(log.epoch, log.val_mean_auc.astype(float), color="#CC79A7", label="mean ROC-AUC")
    b.axvline(best.epoch, color=OTHER, lw=0.6, ls=(0, (3, 2)))
    b.set_xlabel("epoch"); b.set_ylabel("validation score"); b.legend()
    for ax in (a, b):
        ax.xaxis.get_major_locator().set_params(integer=True)
    panel_labels([a, b])
    return fig


def per_class_bars(summaries, emotions, metric="auc", ylabel=None, chance=None):
    """summaries: {run name: metrics summary (test)}. Bars per emotion + mean, 95 % CI error bars.
    Colour = emotion; runs are distinguished by fill (solid, hatched, light, ...)."""
    names = list(summaries)
    cats = list(emotions) + ["mean"]
    n = len(names)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(FULL if n > 1 else SINGLE, 2.2))
    fills = [dict(alpha=1.0, hatch=None), dict(alpha=0.55, hatch="////"), dict(alpha=0.3, hatch=None),
             dict(alpha=0.8, hatch="...")]
    for r, name in enumerate(names):
        s = summaries[name][metric]
        vals = [s["per_class"][e] for e in emotions] + [s["mean"]]
        cis = [s["per_class_ci"][e] for e in emotions] + [s["mean_ci"]]
        x = np.arange(len(cats)) + (r - (n - 1) / 2) * width
        cols = [EMOTION_COLORS[e] for e in emotions] + ["#666666"]
        f = fills[r % len(fills)]
        ax.bar(x, vals, width * 0.92, color=cols, alpha=f["alpha"], hatch=f["hatch"], edgecolor=cols, lw=0.5)
        err = np.array([[v - lo, hi - v] for v, (lo, hi) in zip(vals, cis)]).T
        ax.errorbar(x, vals, yerr=np.nan_to_num(err), fmt="none", ecolor="#222222", elinewidth=0.6, capsize=1.5)
    if chance is not None:
        ax.axhline(chance, color=TEXT_MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(np.arange(len(cats)), cats)
    ax.set_ylabel(ylabel or metric)
    ax.set_ylim(0, 1)
    if n > 1:
        ax.legend(handles=[Patch(facecolor="#888888", alpha=fills[r % 4]["alpha"], hatch=fills[r % 4]["hatch"],
                                 edgecolor="#888888", label=nm) for r, nm in enumerate(names)],
                  ncol=n, loc="upper left")
    return fig


def diff_forest(rows, ref_name):
    """Paired-bootstrap differences (mean metric minus reference) with 95 % CIs."""
    fig, ax = plt.subplots(figsize=(SINGLE, 0.35 + 0.18 * len(rows)))
    labels = [f"{r['run']}: {r['label']}" for r in rows]
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        col = "#0072B2" if r["significant"] else "#888888"
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=col, lw=1.2)
        ax.plot([r["diff"]], [yi], "o", color=col, ms=3)
    ax.axvline(0, color="#333333", lw=0.6)
    ax.set_yticks(y, labels)
    ax.set_xlabel(f"difference to {ref_name} (95 % CI)")
    return fig


# ------------------------------------------------------------------ clips
def example_clips(frames, Y, P, thresholds, emotions, names=None):
    """Isak's example plot, restyled: top = face frame, bottom = per-emotion probabilities.
    Bars in the emotion colour; true labels are filled, the others hollow; the short black tick on
    each bar is that emotion's validation-tuned threshold."""
    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(FULL, 2.9), gridspec_kw={"height_ratios": [1.15, 1]}, squeeze=False)
    x = np.arange(len(emotions))
    for j in range(n):
        axes[0, j].imshow(frames[j]); axes[0, j].set_axis_off()
        if names is not None:
            axes[0, j].text(0.5, -0.04, names[j], transform=axes[0, j].transAxes, ha="center", va="top", fontsize=6,
                            color=TEXT_MUTED)
        ax = axes[1, j]
        for i, e in enumerate(emotions):
            true = Y[j, i] > 0.5
            ax.bar(i, P[j, i], 0.75, color=EMOTION_COLORS[e] if true else "white", edgecolor=EMOTION_COLORS[e], lw=0.8)
            ax.plot([i - 0.38, i + 0.38], [thresholds[e]] * 2, color="#111111", lw=0.8)
        ax.set_ylim(0, 1); ax.set_xticks(x, [e[:3] for e in emotions], rotation=90)
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("probability")
    fig.legend(handles=[Patch(facecolor="#777777", edgecolor="#777777", label="true label"),
                        Patch(facecolor="white", edgecolor="#777777", label="not labelled"),
                        Line2D([], [], color="#111111", label="threshold (validation)")],
               loc="outside lower center", ncol=3)
    panel_labels(axes[0])
    return fig


def tsne_panel(ax, xy, labels, kind="emotion", low_conf=None, emotions=None):
    """labels: emotion names / None (grey 'other') for kind='emotion'; cluster ids for kind='cluster'."""
    labels = np.asarray(labels, dtype=object)
    if kind == "emotion":
        other = np.array([lab is None for lab in labels])
        ax.scatter(xy[other, 0], xy[other, 1], s=4, c=OTHER, lw=0, zorder=1)
        for e in emotions:
            m = labels == e
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=9, c=EMOTION_COLORS[e], marker=EMOTION_MARKERS[e], lw=0.2,
                           edgecolors="white", zorder=2)
    else:
        for k in sorted(set(labels.tolist())):
            m = labels == k
            ax.scatter(xy[m, 0], xy[m, 1], s=9, c=cluster_color(k), marker=cluster_marker(k), lw=0.2,
                       edgecolors="white", zorder=2)
    if low_conf is not None and np.any(low_conf):
        ax.scatter(xy[low_conf, 0], xy[low_conf, 1], s=10, marker="x", c="#111111", lw=0.5, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#CCCCCC")


def tsne_figure(panels, emotions, width=None):
    """panels: list of dict(xy, labels, kind, low_conf=None, note=None)."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(width or (SINGLE if n == 1 else FULL), 2.6 if n > 1 else 2.9),
                             squeeze=False)
    kinds = set()
    for ax, p in zip(axes[0], panels):
        tsne_panel(ax, p["xy"], p["labels"], p.get("kind", "emotion"), p.get("low_conf"), emotions)
        kinds.add(p.get("kind", "emotion"))
        if p.get("note"):
            ax.set_xlabel(p["note"], fontsize=7, color=TEXT_MUTED)
    handles = []
    if "emotion" in kinds:
        handles += [Line2D([], [], ls="", marker=EMOTION_MARKERS[e], color=EMOTION_COLORS[e], label=e) for e in emotions]
        handles.append(Line2D([], [], ls="", marker="o", color=OTHER, label="multi / no label"))
    if "cluster" in kinds:
        ks = sorted({int(k) for p in panels if p.get("kind") == "cluster" for k in p["labels"]})
        handles += [Line2D([], [], ls="", marker=cluster_marker(k), color=cluster_color(k), label=f"cluster {k}")
                    for k in ks]
    if any(p.get("low_conf") is not None and np.any(p["low_conf"]) for p in panels):
        handles.append(Line2D([], [], ls="", marker="x", color="#111111", label="low confidence"))
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 5))
    if n > 1:
        panel_labels(axes[0])
    return fig


def gradcam_grid(items, ncols=5):
    """items: list of dict(frame (S,S,3), cam (S,S), text). One shared colourbar."""
    ncols = max(1, min(ncols, len(items)))
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL * max(ncols, 3) / 5, 1.55 * nrows + 0.3), squeeze=False)
    im = None
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, it in zip(axes.ravel(), items):
        ax.imshow(it["frame"])
        im = ax.imshow(it["cam"], cmap=CAM_CMAP, alpha=0.5, vmin=0, vmax=1)
        ax.text(0.5, -0.03, it["text"], transform=ax.transAxes, ha="center", va="top", fontsize=6)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
        cb.set_label("Grad-CAM (normalised)")
        cb.solids.set(alpha=1)
    return fig


def review_sheet(frames16, probs, emotions, info):
    """4x4 contact sheet of the 16 frames a model saw, with all probabilities as text below."""
    fig, axes = plt.subplots(4, 4, figsize=(SINGLE, SINGLE + 0.45))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(frames16[i]); ax.set_axis_off()
        ax.text(0.03, 0.97, str(i + 1), transform=ax.transAxes, color="white", fontsize=5, va="top")
    order = np.argsort(probs)[::-1]
    txt = "  ".join(f"{emotions[k]} {probs[k]:.2f}" for k in order)
    fig.text(0.5, 0.0, f"{info}\n{txt}", ha="center", va="top", fontsize=6)
    return fig


# ------------------------------------------------------------------ patient videos
def _band(ax, segs, duration):
    for lab, a, b in segs:
        ax.axvspan(a, b, ymin=1.02, ymax=1.09, color=label_color(lab), clip_on=False, lw=0)
        if b - a > duration / 22:
            dark = lab in EMOTION_COLORS or str(lab).startswith("cluster ")
            ax.text((a + b) / 2, 1.055, lab, transform=ax.get_xaxis_transform(), ha="center", va="center",
                    fontsize=5.5, color="white" if dark else "#333333", clip_on=False)


def _time_axis(ax, duration):
    from common.timeline import fmt_time
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_time))
    ax.set_xlim(0, duration)


def timeline_panel(ax, tl, emotions, kind="emotion", min_score=0.5):
    """tl: dict(centre_s, probs_smooth (W,6) or clusters (W,), valid, segments, stride, duration)."""
    c, stride, duration = tl["centre_s"], tl["stride"], tl["duration"]
    for i in np.where(~tl["valid"])[0]:
        ax.axvspan(c[i] - stride / 2, c[i] + stride / 2, color=NO_FACE_COLOR, lw=0, zorder=0)
    if kind == "emotion":
        for i, e in enumerate(emotions):
            ax.plot(c, tl["probs_smooth"][:, i], color=EMOTION_COLORS[e], lw=0.8, label=e, zorder=2)
        ax.axhline(min_score, color=TEXT_MUTED, lw=0.5, ls=(0, (3, 3)), zorder=1)
        ax.set_ylim(0, 1); ax.set_ylabel("probability")
    else:
        cl = np.asarray(tl["clusters"], float)
        ks = sorted({int(k) for k in cl[np.isfinite(cl)]})
        for k in ks:
            m = cl == k
            ax.scatter(c[m], cl[m], s=3, c=cluster_color(k), marker="|", lw=0.8, zorder=2)
        ax.set_yticks(ks, [f"cluster {k}" for k in ks]); ax.set_ylim(-0.6, max(ks + [0]) + 0.6)
    _band(ax, tl["segments"], duration)
    _time_axis(ax, duration)


def timeline_figure(tl, emotions, kind="emotion", min_score=0.5):
    fig, ax = plt.subplots(figsize=(FULL, 1.9))
    timeline_panel(ax, tl, emotions, kind, min_score)
    ax.set_xlabel("time in video (min:s); shaded = no face")
    if kind == "emotion":
        fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper")
    return fig


def stacked_timelines(items, emotions, min_score=0.5):
    """items: list of (name, tl, kind); one row per track, shared time axis."""
    fig, axes = plt.subplots(len(items), 1, figsize=(FULL, 1.45 * len(items) + 0.3), sharex=True, squeeze=False)
    for ax, (name, tl, kind) in zip(axes[:, 0], items):
        timeline_panel(ax, tl, emotions, kind, min_score)
        ax.text(1.01, 0.5, name, transform=ax.transAxes, rotation=270, va="center", fontsize=7)
    axes[-1, 0].set_xlabel("time in video (min:s); shaded = no face")
    emotion_legend(fig, emotions, loc="outside lower center", ncol=6)
    panel_labels(axes[:, 0])
    return fig


def annotated_frames(rows, emotions):
    """rows: list of dict(image RGB, box, probs (6,), time_s). Frame with face box | probability bars."""
    from common.timeline import fmt_time
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(SINGLE * 1.5, 1.45 * n), gridspec_kw={"width_ratios": [1.3, 1]},
                             squeeze=False)
    for (a, b), r in zip(axes, rows):
        a.imshow(r["image"]); a.set_axis_off()
        if r.get("box") is not None and np.isfinite(r["box"]).all():
            x1, y1, x2, y2 = r["box"]
            a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#F0E442", lw=1.0))
        a.text(0.01, 0.99, fmt_time(r["time_s"]), transform=a.transAxes, va="top", fontsize=6, color="white",
               bbox=dict(fc="black", alpha=0.5, lw=0, pad=1))
        p = r["probs"]
        if np.isfinite(p).all():
            b.barh(np.arange(len(emotions)), p, color=[EMOTION_COLORS[e] for e in emotions], height=0.7)
        b.set_yticks(np.arange(len(emotions)), emotions); b.invert_yaxis(); b.set_xlim(0, 1)
    axes[-1, 1].set_xlabel("probability")
    return fig


# ------------------------------------------------------------------ clustering
def k_sweep(ks, silhouettes, inertias, chosen):
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL * 0.7, 1.9))
    a.plot(ks, silhouettes, "o-", color="#0072B2", ms=3)
    b.plot(ks, inertias, "o-", color="#333333", ms=3)
    for ax in (a, b):
        ax.axvline(chosen, color="#D55E00", lw=0.7, ls=(0, (3, 2)))
        ax.set_xlabel("number of clusters k")
        ax.xaxis.get_major_locator().set_params(integer=True)
    a.set_ylabel("silhouette"); b.set_ylabel("inertia (elbow)")
    panel_labels([a, b])
    return fig


def cluster_representatives(rows, n_cols):
    """rows: list of (cluster id, n members, [frames])."""
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(FULL * min(1.0, n_cols / 6), 1.1 * len(rows)), squeeze=False)
    for (k, n, frames), ax_row in zip(rows, axes):
        for j, ax in enumerate(ax_row):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j < len(frames):
                ax.imshow(frames[j])
        ax_row[0].set_ylabel(f"cluster {k}\n(n={n})", fontsize=7, color=cluster_color(k))
    return fig


def cluster_profiles(z, feature_names, clusters, readout, emotions, top=4):
    """(a) mean blendshape z-score per cluster (union of each cluster's top-|z| features);
    (b) P(emotion | cluster) estimated on the training clips."""
    sel = []
    for row in z:
        for j in np.argsort(-np.abs(row))[:top]:
            if j not in sel:
                sel.append(j)
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 0.6 + 0.22 * len(sel)),
                               gridspec_kw={"width_ratios": [max(len(clusters), 2), len(emotions)]})
    lim = float(np.nanmax(np.abs(z[:, sel]))) or 1.0
    im = a.imshow(z[:, sel].T, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    a.set_yticks(range(len(sel)), [feature_names[j] for j in sel], fontsize=6)
    a.set_xticks(range(len(clusters)), [str(k) for k in clusters]); a.set_xlabel("cluster")
    fig.colorbar(im, ax=a, shrink=0.8, pad=0.01).set_label("z-score")
    im2 = b.imshow(readout, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    b.set_xticks(range(len(emotions)), emotions, rotation=90); b.set_yticks(range(len(clusters)), [str(k) for k in clusters])
    b.set_ylabel("cluster")
    fig.colorbar(im2, ax=b, shrink=0.8, pad=0.01).set_label("P(emotion | cluster)")
    panel_labels([a, b])
    return fig


def time_share_bars(shares, order=None):
    """shares: {track name: {label: share}}. Stacked horizontal bars, one per track."""
    names = list(shares)
    labels = order or sorted({lab for s in shares.values() for lab in s})
    fig, ax = plt.subplots(figsize=(FULL, 0.45 + 0.3 * len(names)))
    left = np.zeros(len(names))
    for lab in labels:
        vals = np.array([shares[n].get(lab, 0.0) for n in names])
        ax.barh(np.arange(len(names)), vals, left=left, color=label_color(lab), height=0.6, label=lab,
                edgecolor="white", lw=0.3)
        left += vals
    ax.set_yticks(np.arange(len(names)), names); ax.set_xlim(0, 1); ax.set_xlabel("share of video time")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=min(len(labels), 6))
    return fig


def detection_sheet(rows, col_names):
    """Diagnostic: rows = clips, columns = crop variants (frames RGB or None)."""
    fig, axes = plt.subplots(len(rows), len(col_names), figsize=(1.1 * len(col_names), 1.1 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            ax = axes[i, j]
            ax.set_axis_off()
            if img is not None:
                ax.imshow(img)
            if i == 0:
                ax.text(0.5, 1.03, col_names[j], transform=ax.transAxes, ha="center", va="bottom", fontsize=6)
    return fig
