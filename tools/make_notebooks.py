"""Regenerate the six stage notebooks (run from the repository root: python tools/make_notebooks.py).

The notebooks only hold a title, a short description, one config cell and calls into
common/stages.py. Edit the cells here, not in the .ipynb files, so the six stay consistent.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/LukasStrandberg/Scientific-computing-project---Emotion-detection.git"

COLAB = '''# Colab: mount Drive and run the bootstrap written by 00_setup.ipynb (code + packages). Locally: nothing happens.
import sys
if "google.colab" in sys.modules:
    from google.colab import drive; drive.mount("/content/drive")
    exec(open("/content/drive/MyDrive/emotion_detection_bootstrap.py").read())'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


NOTEBOOKS = {
    "00_setup": [
        md("""
# 00 Setup
Prepares the environment for all other notebooks. **Colab:** mounts Google Drive, gets the code
(clone of `REPO_URL` / `BRANCH`), installs `requirements.txt` and writes a small bootstrap file to
your Drive that the other notebooks run. **Locally:** checks the venv / kernel, packages and GPU.
Both: downloads the two small weight files and lists which checkpoints are still missing.
"""),
        code(f'''
DRIVE_ROOT = None      # Colab: REQUIRED, a Drive folder for data/, checkpoints/, cache/, results/,
                       #        e.g. "/content/drive/MyDrive/<your folder>". Locally: None = the repo folder.
REPO_URL = "{REPO_URL}"
BRANCH = "unified"     # the branch that holds this code
SMOKE_TEST = False     # True: synthetic data + tiny settings (checks that everything runs)
'''),
        code('''
import os, subprocess, sys
if "google.colab" in sys.modules:          # get the code first, so that `common` can be imported
    repo = "/content/emotion-detection-unified"
    if not os.path.isdir(repo + "/common"):
        subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO_URL, repo], check=True)
    os.chdir(repo); sys.path.insert(0, repo)
if SMOKE_TEST:
    os.environ["EMO_SMOKE"] = "1"

from common.env import setup
cfg = setup(DRIVE_ROOT, REPO_URL, BRANCH)
'''),
        md("""
**Next:** put the data and checkpoints where the table above says (layout: README, *Data layout*),
then run `01_preprocess.ipynb`. Every later notebook has a `MODEL` switch:
`'s2d'`, `'maedfer'` or `'clustering'`.
"""),
    ],
    "01_preprocess": [
        md("""
# 01 Preprocess
Builds the shared train / val / test split and the shared face-crop cache (32 uniformly sampled,
face-cropped 224x224 frames per ElderReact clip; resumable; kept as one tar on Drive in Colab).
For `MODEL = 'clustering'` it also extracts the MediaPipe blendshapes from the same crops.
"""),
        code(COLAB),
        code('''
MODEL = "s2d"            # "s2d" | "maedfer" | "clustering" (the crop cache is shared by all three)
RUN_DIAGNOSTIC = False   # True: face-detection diagnostic on a sample of clips (detectors, crops, MediaPipe)
REDO = False             # True: rebuild even if the outputs exist
SMOKE_TEST = False
'''),
        code('''
from common.env import init
from common import data, stages
cfg = init(MODEL, smoke_test=SMOKE_TEST)
'''),
        md("## Split\nOfficial train + dev pooled and re-split 85 / 15 (seed 42); official test untouched."),
        code('''
df = stages.prepare_split(cfg)
data.split_summary(df, cfg.emotions)
'''),
        md("## Face-crop cache\nRetinaFace on every sampled frame, eye-levelled `deepface` crop (config `faces`)."),
        code("stages.build_face_cache(cfg, df, redo=REDO)"),
        md("## Diagnostic (optional)\nDetection rate, clips without a face, box jitter, MediaPipe on crops vs full frames."),
        code('''
if RUN_DIAGNOSTIC:
    display(stages.face_diagnostic(cfg, df))
'''),
        md("## Blendshapes (clustering only)"),
        code('''
if cfg.model == "clustering":
    stages.extract_blendshapes(cfg, df, redo=REDO)
'''),
    ],
    "02_train": [
        md("""
# 02 Train
Fine-tunes the model on ElderReact (S2D, MAE-DFER: best epoch by validation loss, early stopping),
or fits scaler + autoencoder + k-means (clustering). Skipped if the run already exists (then this
notebook only redraws the training figures); `RESUME = True` continues from `best.pth`.
"""),
        code(COLAB),
        code('''
MODEL = "s2d"     # "s2d" | "maedfer" | "clustering"
RESUME = False    # supervised: continue fine-tuning from best.pth (new optimiser / schedule)
REDO = False      # True: train again from scratch (overwrites the run)
SMOKE_TEST = False
'''),
        code('''
from common.env import init
from common import stages
cfg = init(MODEL, smoke_test=SMOKE_TEST)
'''),
        code("stages.train(cfg, resume=RESUME, redo=REDO)"),
    ],
    "03_evaluate": [
        md("""
# 03 Evaluate
For every run of the model (S2D / MAE-DFER: `baseline` = zero-shot source model, `finetuned`;
clustering: `ae_kmeans`): validation predictions -> per-emotion thresholds -> test metrics with
95 % bootstrap CIs, the run files (`results/<model>/<run>/`) and the standard figures.
"""),
        code(COLAB),
        code('''
MODEL = "s2d"     # "s2d" | "maedfer" | "clustering"
RUNS = None       # None = all runs of the model, or e.g. ["finetuned"]
REDO = False
SMOKE_TEST = False
'''),
        code('''
from common.env import init
from common import stages
cfg = init(MODEL, smoke_test=SMOKE_TEST)
'''),
        code("stages.evaluate(cfg, RUNS, redo=REDO)"),
    ],
    "04_patient_videos": [
        md("""
# 04 Patient videos
Cuts each long patient video into windows (default preset `s2d`: 3 s every 1 s, smoothed over 5
windows), face-crops them like the training data and scores them with the chosen run. Writes the
timeline, segments and summary per video, plus timeline, annotated frames, review sheets, Grad-CAM
(supervised) and t-SNE figures.
"""),
        code(COLAB),
        code('''
MODEL = "s2d"            # "s2d" | "maedfer" | "clustering"
RUN = None               # None = the trained run (finetuned / ae_kmeans); "baseline" also works
WINDOW_PRESET = "s2d"    # "s2d" (3 s every 1 s, smoothed) | "maedfer" (1 s every 0.8 s, unsmoothed)
REDO = False
SMOKE_TEST = False
'''),
        code('''
from common.env import init
from common import stages
cfg = init(MODEL, smoke_test=SMOKE_TEST, patient__preset=WINDOW_PRESET)
'''),
        code("stages.patient_videos(cfg, run=RUN, redo=REDO)"),
    ],
    "05_compare": [
        md("""
# 05 Compare
Cross-model comparison, read only from the run files: test metrics with 95 % CIs, paired-bootstrap
differences to a reference run, side-by-side per-emotion bars and t-SNE, and per patient video the
stacked timelines and time shares of all runs.
"""),
        code(COLAB),
        code('''
RUNS = None              # None = every run in results/, or e.g. ["s2d/finetuned", "maedfer/finetuned", "clustering/ae_kmeans"]
REFERENCE = None         # run the differences are computed against (default: the first run)
WINDOW_PRESET = "s2d"    # which patient timelines to compare
SMOKE_TEST = False
'''),
        code('''
from common.env import init
from common import stages
cfg = init(None, smoke_test=SMOKE_TEST)
'''),
        code('''
table, differences = stages.compare(cfg, RUNS, REFERENCE, WINDOW_PRESET)
display(table)
differences
'''),
    ],
}


def main():
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        nbf.write(nb, ROOT / f"{name}.ipynb")
        print("wrote", f"{name}.ipynb")


if __name__ == "__main__":
    main()
