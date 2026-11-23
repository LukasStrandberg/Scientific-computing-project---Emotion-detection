# Emotion detection in videos of Alzheimer's patients: unified codebase

Three tracks detect emotions, and when they change, in long videos of Alzheimer's patients, after
learning from **ElderReact** (1,323 clips of older adults; 6 binary emotions: anger, disgust, fear,
happiness, sadness, surprise; multi-label, no neutral):

| `MODEL` | Track | Idea | Details |
|---|---|---|---|
| `s2d` | S2D (Chen et al., TAFFC 2024) | landmark-aware ViT, MAFW-trained, fine-tuned on ElderReact | [s2d/README.md](s2d/README.md) |
| `maedfer` | MAE-DFER (Sun et al., ACM MM 2023) | self-supervised video ViT (LGI-Former), DFEW-trained, linear probe on ElderReact | [mae_dfer/README.md](mae_dfer/README.md) |
| `clustering` | unsupervised | MediaPipe blendshapes → autoencoder → k-means | [clustering/README.md](clustering/README.md) |

(The MAE-DFER folder is `mae_dfer/`, not `mae-dfer/`: Python cannot import a package name with a hyphen.)

All tracks share one split, one face detector and crop cache, one frame-sampling rule, one training
protocol (supervised tracks), one set of metrics with bootstrap CIs, one output format, the same
patient-video windows and one figure style.

## Notebook order

| Notebook | What it does | Re-running |
|---|---|---|
| `00_setup.ipynb` | Colab: Drive, code, packages, bootstrap. Local: checks. Both: small weights + checklist of missing checkpoints | always fine |
| `01_preprocess.ipynb` | shared split, shared face-crop cache (+ blendshapes for clustering), optional face-detection diagnostic | resumable; finished clips skipped |
| `02_train.ipynb` | fine-tune (S2D, MAE-DFER) / fit scaler + autoencoder + k-means (clustering) | skipped if done; `RESUME`, `REDO` |
| `03_evaluate.ipynb` | val → thresholds → test metrics with 95 % CIs, run files, standard figures (every run) | skipped if done; `REDO` |
| `04_patient_videos.ipynb` | windows over the patient videos: timeline, segments, summary, figures | skipped per video if done |
| `05_compare.ipynb` | all runs side by side: CI tables, paired-bootstrap differences, figures, stacked timelines | always fine |

Each notebook starts with a short description and **one config cell**: `MODEL = 's2d' | 'maedfer' |
'clustering'` plus the few settings worth changing. The remaining cells call functions in
`common/stages.py`. Everything else is in [config.yaml](config.yaml). Run 01-04 once per `MODEL`
(the face cache from 01 is shared, so later models skip it), then 05.

The same stages run from a terminal: `python tools/run.py {preprocess,train,evaluate,patients,compare} --model s2d`
(see `python tools/run.py -h`; `--set section.key=value` overrides any config value).

## Setup

### Local (Windows PowerShell)
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 1) PyTorch FIRST, with the command from https://pytorch.org/get-started/locally
#    (Stable, Windows, Pip, Python, your CUDA), e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2) everything else
pip install -r requirements.txt
# 3) register the venv as a notebook kernel and select it in VS Code / Jupyter
python -m ipykernel install --user --name emotion-unified --display-name "Python (emotion-unified)"
```
If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### Local (Linux / macOS)
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # from pytorch.org
pip install -r requirements.txt
python -m ipykernel install --user --name emotion-unified --display-name "Python (emotion-unified)"
```
Open the notebooks from the repository folder (the kernel's working directory must be the repo root).
Locally the data root is the repository folder; set the environment variable `EMO_ROOT` (or
`DRIVE_ROOT` in `00_setup`) to keep data elsewhere.

### Google Colab
1. Push this code to a branch of the project repository (the notebooks clone `REPO_URL` at `BRANCH`).
2. Create a folder on your Drive for the data (any name). Put the data and checkpoints in it, laid out
   as below. Upload the face-crop cache tar too if another machine already built it.
3. Open `00_setup.ipynb` in Colab (File → Open notebook → GitHub), choose a GPU runtime and set
   **`DRIVE_ROOT`** (required, no default; e.g. `"/content/drive/MyDrive/<your folder>"`), `REPO_URL`
   and `BRANCH`. Run it.
4. Open the other notebooks. Their first cell mounts Drive and runs the bootstrap that `00_setup`
   wrote to `MyDrive/emotion_detection_bootstrap.py` (clone + `pip install`, once per runtime).

Colab's preinstalled PyTorch is used as is (`requirements.txt` does not list torch). The face cache
is built on the fast local disk (`/content/emo_cache`) and kept on Drive as **one tar file**
(`<DRIVE_ROOT>/cache/<cache tag>.tar`). Later sessions extract it instead of rebuilding.

### Pinned versions and why
Only `timm==0.6.13` is pinned. S2D's model code needs timm 0.6.x. MAE-DFER (written for 0.4.12)
runs unchanged on 0.6.13 because its model function is called directly instead of through
`timm.create_model` (see `third_party/mae_dfer/NOTICE.md`). Tested with Python 3.13, torch 2.15,
numpy 2.5, mediapipe 1.2.3, batch-face 1.5.4 and ultralytics 8.4. Upstream model code is
**vendored, unmodified**, under `third_party/` with its license (S2D: Apache-2.0, MAE-DFER: MIT).

## Data layout
All paths are relative to the data root (`DRIVE_ROOT` on Colab, the repo folder locally):
```
data/
  ElderReact_Data/
    train_labels.txt  dev_labels.txt  test_labels.txt     # from github.com/Mayer123/ElderReact (Annotations/)
    ElderReact_train/*.mp4  ElderReact_dev/*.mp4  ElderReact_test/*.mp4   # any sub-folders work
  AlzheimerPatient_Data/                                  # the patient videos (.mp4/.mov/...)
checkpoints/
  s2d/mafw_split01.pth           # huggingface.co/cyinen/S2D: MAFW/without_oversampler/split01_checkpoint-best.pth
  mae_dfer/dfew_fold1.pth        # MAE-DFER README: DFEW fine-tuned checkpoint, fold 1 (Google Drive link)
  face_landmarker.task           # downloaded by 00_setup
  yolov8n-face.pt                # optional, only for faces.detector: yolo
cache/                           # face crops, blendshapes (git-ignored)
results/                         # every output (git-ignored)
```
MobileFaceNet (S2D's landmark branch, 12 MB) is downloaded by `00_setup` into
`third_party/s2d/landmark/`. RetinaFace weights are downloaded by batch-face on first use.
Optional extras are listed in the model READMEs. `00_setup` prints a table of what is present.

The **split** is shipped in `splits/elderreact_pooled_seed42.csv` (824 train / 146 val / 353
official test). `01_preprocess` checks it against your label files.

## Output schema
Every run writes the same files. `05_compare` reads only these.
```
results/<model>/<run>/                  model ∈ {s2d, maedfer, clustering}; run ∈ {baseline, finetuned, ae_kmeans}
  config.json       resolved settings, split file, cache tag, SHA-256 of the source checkpoint
  metrics.json      {"val": ..., "test": ...}: per emotion and mean for auc, ap, f1 (val thresholds), f1_05,
                    each with 95 % bootstrap CI; single-label WAR/UAR; clustering: silhouette, NMI, ARI, purity
  thresholds.json   per-emotion F1 thresholds tuned on the validation split
  predictions.csv   video, split, true_<emotion> x6, prob_<emotion> x6 [, cluster]
  features.npz      features (N, D), video, split, labels, probs [, cluster]
  log.csv           training log; best.pth (supervised) / model.pt (clustering)
  figures/          *.pdf + *.png
  patients/<preset>/<video>_timeline.csv   start_s, end_s, centre_s, face_share, prob_<emotion> x6 [, cluster], predicted
  patients/<preset>/<video>_segments.csv   label, start_s, end_s (stretches with the same dominant label)
  patients/<preset>/<video>_summary.json   time share per label, changes per minute, mean segment length
  patients/<preset>/figures/               timeline, annotated frames, Grad-CAM grid, t-SNE, review/ sheets
results/<model>/figures/                 baseline vs fine-tuned side by side
results/compare/                         tables (CSV) + figures from 05_compare
results/diagnostics/                     face-detection diagnostic from 01_preprocess
```

## Shared protocol (summary)
- **Split:** official train + dev pooled, 85 / 15 re-split with seed 42 (MAE-DFER track's rule); official test untouched.
- **Faces:** RetinaFace-R50 on every sampled frame, largest face, eyes levelled, `deepface` crop (tight box, black padding), 224 px. Frames without a face reuse the nearest box. Clips without any face fall back to a centred crop and are flagged, so every track scores the same 353 test clips.
- **Frames:** 32 uniform bin-centre frames per clip. Models see 16: training takes a random one of each pair plus a horizontal flip; evaluation averages the even- and odd-frame views. Only input size and normalisation differ per model (S2D 224, MAE-DFER 160, ImageNet mean/std).
- **Training (S2D, MAE-DFER):** seed 42, BCE-with-logits with `pos_weight = clip(neg/pos, 1, 10)`, AdamW (wd 0.05), 1 warm-up epoch + cosine, AMP, gradient clipping 1.0, best epoch by validation loss, early stopping after 8 epochs.
- **Metrics:** ROC-AUC, AP / mAP, F1 with validation-tuned thresholds, F1@0.5; 95 % bootstrap CIs (2,000 resamples); paired-bootstrap differences in the comparison.
- **Patient videos:** default 3 s windows every 1 s, smoothed over 5 windows (`s2d` preset); `maedfer` preset: 1 s every 0.8 s, unsmoothed. Same windows for every track.
- **Figures:** two-column LaTeX sizes (3.4 / 7.0 in), serif 8–9 pt, fixed Okabe–Ito colour per emotion, no in-figure titles (`plot.titles: true` adds them), PDF + 300 dpi PNG.

## Tests
```bash
python -m pytest -q                 # unit tests (~1 min on a CPU, most of it the two model tests)
python tools/run_smoke.py --fresh   # every notebook for every MODEL on synthetic data (~20 min on a CPU)
```
`SMOKE_TEST = True` in a notebook's config cell (or `smoke_test: true` in config.yaml, or env
`EMO_SMOKE=1`) uses tiny synthetic data under `cache/_smoke/`, a stub face detector, stub
blendshapes and randomly initialised models with the real architectures. The pipeline runs
end-to-end, but the numbers mean nothing.

`tools/make_notebooks.py` regenerates the six notebooks. Change cells there, not in the `.ipynb`
files.

## Repository layout
```
common/       shared code: env, config, data, faces, sampling, cache, diagnostics, metrics, training,
              outputs, plotting (+ emotion.mplstyle), tsne, gradcam, timeline, tracks, stages, smoke
s2d/ mae_dfer/ clustering/   model-specific code + README each
third_party/  vendored upstream model code (unmodified, with licenses)
splits/       the shared split
tests/        pytest unit tests (+ reference state-dict key lists generated from upstream)
tools/        run.py (CLI), run_smoke.py, make_notebooks.py
```
