"""t-SNE "cluster plot" of S2D clip features, in the style of Fig. 7 of the S2D paper.

Takes one or more features.npz files written by evaluate.py and draws one panel per file,
side by side, with the same colour AND marker shape per emotion in every panel (colour
alone is not enough to tell six emotions apart for colour-blind readers). Each emotion's
cluster is also labelled with its name.

Which clips are coloured
  * MAFW (single label): the 6 emotions shared with ElderReact + neutral are coloured;
    the 4 rare MAFW emotions are drawn as small grey "other" points.
  * ElderReact (multi-label): clips with exactly one emotion are coloured; clips with
    several emotions or none are drawn as small grey "other" points.
The t-SNE itself always uses ALL clips.

Under each panel the silhouette score of the coloured clips is printed (cosine distance,
computed on the original 768-d features, not on the 2-D picture). Higher = emotions form
more separate clusters; it gives the visual impression a number for the report.

Example - before vs after ElderReact fine-tuning:
    python s2d/plot_tsne.py ^
        --inputs results/er_zeroshot_uni_test/features.npz results/er_ft_test/features.npz ^
        --titles "Before (MAFW model)" "After ElderReact fine-tuning" ^
        --out results/tsne_elderreact_before_after.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

# Fixed emotion -> (colour, marker). Colours: categorical slots 1-7 of the dataviz
# reference palette, in fixed order; same mapping in every panel and every plot.
STYLE = {
    "anger":     ("#e34948", "o"),
    "disgust":   ("#1baf7a", "s"),
    "fear":      ("#6250d6", "^"),
    "happiness": ("#eda100", "D"),
    "sadness":   ("#2a78d6", "v"),
    "surprise":  ("#eb6834", "P"),
    "neutral":   ("#8a8a85", "X"),
}
OTHER_COLOR = "#d3d2cc"
TEXT = "#2b2b29"
MUTED = "#6b6a64"


def clip_classes(npz):
    """Return list with an emotion name (coloured) or None (grey) for every clip."""
    names = [str(c) for c in npz["class_names"]]
    labels = npz["labels"]
    if labels.shape[1] == 1:  # single-label (MAFW)
        out = [names[int(v)] for v in labels[:, 0]]
    else:  # multi-hot (ElderReact)
        out = [names[int(np.argmax(r))] if r.sum() == 1 else None for r in labels]
    return [c if c in STYLE else None for c in out]


def embed(features, seed, perplexity):
    x = features / np.linalg.norm(features, axis=1, keepdims=True)
    x = PCA(n_components=min(50, x.shape[1], len(x) - 1), random_state=seed).fit_transform(x)
    return TSNE(n_components=2, perplexity=min(perplexity, (len(x) - 1) / 3), init="pca",
                random_state=seed).fit_transform(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="features.npz files from evaluate.py")
    ap.add_argument("--titles", nargs="+", help="one title per input")
    ap.add_argument("--out", required=True, help="output image (.png, or .pdf/.svg for the report)")
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    titles = args.titles or [Path(p).parent.name for p in args.inputs]

    n = len(args.inputs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.6), squeeze=False)
    present = []
    for ax, path, title in zip(axes[0], args.inputs, titles):
        d = np.load(path)
        feats = d["features"].astype(np.float64)
        cls = clip_classes(d)
        xy = embed(feats, args.seed, args.perplexity)

        coloured = np.array([c is not None for c in cls])
        sil = (silhouette_score(feats[coloured], [c for c in cls if c], metric="cosine")
               if len({c for c in cls if c}) > 1 else float("nan"))

        labels_at = []
        ax.scatter(xy[~coloured, 0], xy[~coloured, 1], s=10, c=OTHER_COLOR, linewidths=0, zorder=1)
        for emo, (col, mk) in STYLE.items():
            m = np.array([c == emo for c in cls])
            if not m.any():
                continue
            if emo not in present:
                present.append(emo)
            ax.scatter(xy[m, 0], xy[m, 1], s=26, c=col, marker=mk, edgecolors="white",
                       linewidths=0.6, zorder=2)
            if m.sum() >= 10:  # direct label at the cluster's median point (larger clusters only)
                labels_at.append([emo, *np.median(xy[m], axis=0)])
        # push labels apart vertically if they would overlap
        span = np.ptp(xy, axis=0)
        labels_at.sort(key=lambda t: t[2])
        for i in range(1, len(labels_at)):
            for j in range(i):
                dx = abs(labels_at[i][1] - labels_at[j][1]) / span[0]
                dy = (labels_at[i][2] - labels_at[j][2]) / span[1]
                if dx < 0.14 and dy < 0.05:
                    labels_at[i][2] = labels_at[j][2] + 0.05 * span[1]
        for emo, cx, cy in labels_at:
            ax.annotate(emo, (cx, cy), fontsize=9, color=TEXT, ha="center", va="center", zorder=3,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.set_title(title, fontsize=12, color=TEXT, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#e4e3dd")
        ax.set_xlabel(f"{len(cls)} clips, {int(coloured.sum())} coloured  ·  silhouette {sil:.3f}",
                      fontsize=9, color=MUTED)
        print(f"{title}: {len(cls)} clips, {int(coloured.sum())} coloured, silhouette (cosine) = {sil:.4f}")

    handles = [plt.Line2D([], [], ls="", marker=STYLE[e][1], color=STYLE[e][0], markeredgecolor="white",
                          markersize=8, label=e) for e in present]
    handles.append(plt.Line2D([], [], ls="", marker="o", color=OTHER_COLOR, markersize=6,
                              label="other (several / no emotion, or rare class)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
