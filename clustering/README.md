# Clustering track (unsupervised)

**Method.** Each clip (or patient-video window) is described by facial **blendshapes** from MediaPipe
FaceLandmarker: 52 action-unit-like scores in [0, 1] per frame (brow raise, smile, jaw open, ...).
Per clip they are summarised as [mean, max, std] over the frames with a detected face (156-d). A
StandardScaler and a small **autoencoder** (156 → 64 → 32 → 8 → 32 → 64 → 156, MSE) compress them,
and **k-means** clusters the 8-d latent vectors. No labels are used for fitting. The labels are used
afterwards, to describe and score the clusters.

> C. Lugaresi et al. *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172, 2019;
> MediaPipe Face Landmarker (blendshape model), Google AI Edge.

**Code and license.** No upstream model code; MediaPipe is a pip dependency (Apache-2.0). The logic
comes from the `clustering` branch (`video_test.ipynb`, `concatenate_video.ipynb`), translated and
fixed (see "Changes" below).

## Weights
| File | Source | Path |
|---|---|---|
| `face_landmarker.task` (float16, v1) | storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task (auto-download by `00_setup`) | `checkpoints/face_landmarker.task` |

## Run `ae_kmeans` (fit in `02_train`, on the training split only)
1. Blendshapes for the 32 shared face crops of every clip (`01_preprocess`, saved under `cache/blendshapes/`).
2. 156-d clip vectors; clips where MediaPipe finds no face in any frame are left out of the fit.
3. StandardScaler → autoencoder (Adam, lr 1e-3, wd 1e-5, 300 epochs, batch 256, seed 42) → 8-d latent.
4. k-means (`n_init=10`, seed 42) with **k = the best silhouette over k = 2…12** on the training clips (`clustering.k` can fix k).
5. **Read-out** P(emotion | cluster) = (positives + 1) / (members + 2) on the training clips. A cluster thus becomes 6 scores, so AUC / mAP / F1 with CIs are computed exactly as for S2D and MAE-DFER and appear in the same comparison table.

`--set clustering.embedding=pca --set clustering.run=pca_kmeans` replaces the autoencoder with PCA (8 components).

## Metrics (in `metrics.json`)
- The shared multi-label metrics on the read-out scores (val thresholds, test CIs).
- `clustering`, for val and test: silhouette (latent space, all clips with a face) and **NMI, ARI and purity against the emotion label on clips with exactly one emotion**. Multi-label and unlabelled clips are excluded from these three because they have no single class; their number is reported.
- Patient summary: time share per cluster, cluster changes per minute, mean segment length (majority vote over the preset's smoothing window instead of a moving average).

## Preprocessing: decisions and deviations
- Frames: the **same 32 uniformly sampled face crops** as the supervised tracks (shared RetinaFace cache), MediaPipe in IMAGE mode per frame. The old notebooks used full frames every 30th frame (with a stepping bug, below). Run `01_preprocess` with `RUN_DIAGNOSTIC = True` on real data: it reports how often MediaPipe finds a face on the deepface crop, the square crop and the full frame. If the crops are clearly worse, set `clustering.source: full_frame` (same frame indices, whole frames).
- Patient windows: the same windows as the other tracks. Blendshapes are computed once per grid frame and aggregated per window.

## Figures
t-SNE of the latent space coloured by cluster and by emotion (single-emotion clips), silhouette /
elbow sweep, autoencoder loss, representative frames per cluster (the test clips closest to each
centroid), per-cluster blendshape profile (z-scores) with the P(emotion | cluster) read-out (this
replaces Grad-CAM), and the cluster timeline for every patient video.

## Changes from the `clustering` branch
- **Bug fixed:** `concatenate_video.ipynb` and `video_test.ipynb` did `frame_idx += step` but read frames sequentially with `cap.read()`. They processed the first N consecutive frames and labelled them with the wrong frame numbers. Frames are now chosen explicitly.
- The k proposed by silhouette was overridden by a hard-coded `k = 6`, and the representative frames came from k-means on the t-SNE coordinates. Now one k-means on the latent space is used for everything; k comes from silhouette unless configured.
- Aggregation and autoencoder now run on ElderReact (fit on train), not on a single test video. Swedish comments translated. The plotly scatter is dropped.

## CLI
```bash
python tools/run.py preprocess --model clustering     # face cache (shared) + blendshapes
python tools/run.py train      --model clustering
python tools/run.py evaluate   --model clustering
python tools/run.py patients   --model clustering
```

## Previous results (old protocol)
There are none to report. The branch explored one test video: on 1,000 frames the silhouette sweep
of the autoencoder latent proposed k = 3, and k = 6 was used. ElderReact was not scored.
