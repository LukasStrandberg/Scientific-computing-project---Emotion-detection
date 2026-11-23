"""t-SNE of clip / window features (same settings for every track).

Features are optionally L2-normalised (deep features), reduced to at most 50 PCA components and
embedded with t-SNE (PCA init, perplexity min(30, (n-1)/3), fixed seed). The silhouette score of
the coloured points is computed on the ORIGINAL features (cosine distance), not on the 2-D picture.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def embed(features, seed=42, perplexity=30.0, l2=True):
    x = np.asarray(features, dtype=np.float64)
    if l2:
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    if len(x) < 4:
        return np.zeros((len(x), 2))
    x = PCA(n_components=min(50, x.shape[1], len(x) - 1), random_state=seed).fit_transform(x)
    return TSNE(n_components=2, perplexity=min(perplexity, (len(x) - 1) / 3), init="pca",
                learning_rate="auto", random_state=seed).fit_transform(x)


def single_emotion_labels(Y, emotions):
    """Emotion name for clips with exactly one label, None for multi-label / unlabelled clips."""
    return [emotions[int(np.argmax(r))] if r.sum() == 1 else None for r in np.asarray(Y)]


def fmt(value):
    return "n/a" if not np.isfinite(value) else f"{value:.3f}"


def silhouette(features, labels, metric="cosine"):
    labels = np.asarray(labels, dtype=object)
    keep = np.array([lab is not None for lab in labels])
    if keep.sum() < 3 or len(set(labels[keep].tolist())) < 2 or len(set(labels[keep].tolist())) >= keep.sum():
        return float("nan")
    return float(silhouette_score(np.asarray(features)[keep], labels[keep].astype(str), metric=metric))
