# S2D track

Our reproduction of **S2D** ("From Static to Dynamic", Chen et al., TAFFC 2024) for
MAFW → ElderReact → Alzheimer patient videos. The plan and all decisions are in the
claude.ai Project doc `claude/S2D_pipeline_plan.md`.

We reuse the official code (github.com/MSA-LMC/S2D) unchanged where possible and only
add our own scripts in this folder.

**Easiest way to run everything: the notebook `s2d/S2D_pipeline.ipynb`** - works in Google
Colab (data on Google Drive, instructions at the top of the notebook) and locally with the
`venv-s2d` kernel. The sections below are the same steps as terminal commands.

## 1. One-time setup (Windows, plain venv, no conda)

From the repo root, in PowerShell:

```powershell
# separate environment for S2D, next to the existing venv
py -m venv venv-s2d
.\venv-s2d\Scripts\Activate.ps1

# PyTorch with CUDA: copy the exact command from https://pytorch.org/get-started/locally
# (choose: Stable, Windows, Pip, Python, the newest CUDA). It looks like:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # should print True + RTX 3060

pip install -r s2d/requirements-s2d.txt

# official S2D code (git-ignored, not committed to our repo)
git clone https://github.com/MSA-LMC/S2D.git external/S2D
```

Checkpoints: download from https://huggingface.co/cyinen/S2D into `external/S2D/ckpts/`:
- `ckpts/pretrain-AffectNet-7.pth` (static image model, starting point for training)
- `ckpts/MAFW/without_oversampler/split01_checkpoint-best.pth` … `split05_…` (S2D trained on MAFW, one per fold;
  each may only be tested on its own fold's test set). We use the *without_oversampler* versions.

On Colab the same steps work in a notebook cell (`!pip install ...`, `!git clone ...`);
Colab already has PyTorch with CUDA installed.

## 2. Data preparation

All outputs go to `data/processed/` (inside `data/`, so git-ignored).

```powershell
# label files in the format S2D reads (fast, a few seconds)
python s2d/make_annotations.py

# face crops, smoke test on the first 50 clips of the MAFW fold-1 test set
python s2d/preprocess_faces.py --videos data/MFAW_Data/clips --out data/processed/MAFW/faces_224 --only-csv data/processed/MAFW/anno/set_1/test.csv --limit 50

# then everything (can be stopped and restarted: finished clips are skipped)
python s2d/preprocess_faces.py --videos data/MFAW_Data/clips --out data/processed/MAFW/faces_224
python s2d/preprocess_faces.py --videos data/ElderReact_Data/ElderReact_train data/ElderReact_Data/ElderReact_dev data/ElderReact_Data/ElderReact_test --out data/processed/ElderReact/faces_224
```

`faces_224/_report.csv` lists, per clip, how many frames had no detected face
(those reuse the nearest detected face) and clips where no face was found at all.

## 3. Smoke test: released S2D model on MAFW fold 1

```powershell
python s2d/evaluate.py --ckpt external/S2D/ckpts/MAFW/without_oversampler/split01_checkpoint-best.pth --anno data/processed/MAFW/anno/set_1/test.csv --frames data/processed/MAFW/faces_224 --out results/smoke_mafw_fold1 --limit 50
```

Clips without face crops are skipped, so `--limit 50` together with the 50 cropped clips above
evaluates exactly those. Prints UAR/WAR; writes `predictions.csv`, `features.npz` (for t-SNE) and
`metrics.txt` to the `--out` folder. 50 clips is too few for a meaningful score; it checks that
everything runs and gives a rough sense of accuracy (paper: about 56% WAR on the full test set).
If you get "CUDA out of memory", add `--batch 2`.

Other options of `evaluate.py` (see the top of the file):
- `--sampling uniform` spreads each 16-frame window over the whole clip instead of 16 consecutive frames.
- `--mode er-zeroshot` runs the MAFW model on ElderReact and scores the 6 shared emotions
  (per-emotion AUC and F1, plus accuracy on clips with exactly one emotion). Run it on `dev.csv` first;
  that writes `thresholds.json`, which you pass to the test run with `--thresholds`.

## 4. ElderReact fine-tuning

Default protocol (aligned with the MAE-DFER track): official train + dev are pooled and
re-split 85 / 15 % into train / validation (fixed seed 42, lists written to the output
folder; `--split-file` takes an existing validation list instead, `--split official` uses
the files as given), best epoch by validation loss, early stopping after 8 epochs without
improvement. The official test set is never used for training or selection.

```powershell
python s2d/train_elderreact.py --out results/er_ft_pooled --lr 5e-5 --warmup 1
python s2d/evaluate.py --mode er --sampling uniform --ckpt results/er_ft_pooled/best.pth --anno results/er_ft_pooled/val_split.csv --frames data/processed/ElderReact/faces_224 --out results/er_ft_pooled_val
python s2d/evaluate.py --mode er --sampling uniform --ckpt results/er_ft_pooled/best.pth --anno data/processed/ElderReact/anno/test.csv --frames data/processed/ElderReact/faces_224 --out results/er_ft_pooled_test --thresholds results/er_ft_pooled_val/thresholds.json
```

Earlier runs on the official split (before aligning with the MAE-DFER track):

```powershell
python s2d/train_elderreact.py --out results/er_ft --split official --select auc --patience 0   # ~2.5 h on the 3060
python s2d/evaluate.py --mode er --sampling uniform --ckpt results/er_ft/best.pth --anno data/processed/ElderReact/anno/dev.csv --frames data/processed/ElderReact/faces_224 --out results/er_ft_dev
python s2d/evaluate.py --mode er --sampling uniform --ckpt results/er_ft/best.pth --anno data/processed/ElderReact/anno/test.csv --frames data/processed/ElderReact/faces_224 --out results/er_ft_test --thresholds results/er_ft_dev/thresholds.json
```

## 5. Figures

```powershell
# t-SNE cluster plot, before vs after ElderReact fine-tuning (CPU, ~15 s)
python s2d/plot_tsne.py --inputs results/er_zeroshot_uni_test/features.npz results/er_ft_test/features.npz --titles "Before (MAFW model)" "After ElderReact fine-tuning" --out results/tsne_elderreact.png

# 3 frames (25/50/75 %) with face box and predicted emotions, per video
python s2d/annotate_video.py --mode er --ckpt results/er_ft_short/best.pth --video "data/AlzheimerPatient_Data/*.mp4" --out results/annotated_er

# emotion-over-time timeline, both models. --preset s2d (default): 3 s windows every 1 s, smoothed;
# --preset maedfer: 1 s windows every 0.8 s, unsmoothed (same as the MAE-DFER track)
python s2d/timeline_video.py --video "data/AlzheimerPatient_Data/*.mp4" --er-ckpt results/er_ft_short/best.pth --mafw-ckpt external/S2D/ckpts/MAFW/without_oversampler/split01_checkpoint-best.pth --out results/timeline

# compare ElderReact models on the test set with 95 % bootstrap intervals (AUC, mAP, F1 tuned, F1@0.5);
# --runs also takes CSVs in the shared layout described at the top of compare_runs.py (e.g. MAE-DFER predictions)
python s2d/compare_runs.py --runs results/er_zeroshot_uni_test results/er_ft_test results/er_ft_short_test results/er_ft_head_test --names "before (MAFW)" "fine-tuned 30 ep" "fine-tuned 10 ep" "classifier only" --out results/compare_elderreact.txt
```

### What the scripts produce
| Output | Content |
|---|---|
| `data/processed/MAFW/anno/set_k/{train,test}.csv` | `video_name,labels`, 11 classes in official S2D order (0 anger, 1 anxiety, 2 contempt, 3 disappointment, 4 disgust, 5 fear, 6 happiness, 7 helplessness, 8 neutral, 9 sadness, 10 surprise). Verified identical to the CSVs shipped with S2D. |
| `data/processed/ElderReact/anno/{train,dev,test}.csv` | `video_name,anger,disgust,fear,happiness,sadness,surprise` as 0/1 (multi-label) |
| `data/processed/<dataset>/faces_224/<clip>/00001.jpg …` | aligned 224×224 crop of the largest face, every frame |

## Status
- [x] Label CSVs (MAFW 5 folds, ElderReact 3 splits)
- [x] Face cropping (tested on 3 MAFW + 2 ElderReact clips)
- [x] Official S2D model builds and runs with PyTorch 2.x + timm 0.6.13
- [x] Evaluation script (official 2-window test protocol, saves features for t-SNE); checkpoint keys verified to match the model
- [x] Smoke test + full fold-1 test of the released MAFW model: 44.21 WAR / 30.39 UAR (deepface-style crops)
- [x] Frame-sampling test: uniform 48.18 WAR / 33.59 UAR on MAFW fold 1 (vs 44.21 / 30.39 consecutive)
- [x] ElderReact zero-shot (uniform): test mean AUC 63.69, mean F1 40.81
- [x] t-SNE plot script (`plot_tsne.py`), tested on real features
- [x] 3-frame annotation script (`annotate_video.py`), tested on sample clips
- [x] ElderReact fine-tuning: 30 ep / 10 ep (chosen: `results/er_ft_short`) / classifier only; test mean AUC 63.7 -> 66.4
- [x] t-SNE before/after ElderReact (silhouette -0.118 -> -0.054, 30-epoch model)
- [x] Patient videos: 3 annotated frames
- [x] Patient videos: timeline (`timeline_video.py`), presets for both window settings
- [x] Aligned with the MAE-DFER track: pooled split, validation-loss selection + early stopping, mAP / F1@0.5
- [x] Notebook for Colab and local runs (`S2D_pipeline.ipynb`)
- [ ] Re-train with the pooled split (exact MAE-DFER validation list when available)
