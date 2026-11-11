# S2D track

Our reproduction of **S2D** ("From Static to Dynamic", Chen et al., TAFFC 2024) for
MAFW → ElderReact → Alzheimer patient videos. The plan and all decisions are in the
claude.ai Project doc `claude/S2D_pipeline_plan.md`.

We reuse the official code (github.com/MSA-LMC/S2D) unchanged where possible and only
add our own scripts in this folder.

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
- [ ] Smoke test: released MAFW checkpoint on a MAFW subset (local 3060)
- [ ] ElderReact multi-label fine-tuning
- [ ] t-SNE before/after ElderReact
- [ ] Patient videos: sliding windows, timeline, 3 annotated frames
