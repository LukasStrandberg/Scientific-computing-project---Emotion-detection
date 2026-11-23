# Vendored: S2D (From Static to Dynamic)

- Source: https://github.com/MSA-LMC/S2D at commit `e1b5e5dfb57ab15f70103c933b346bfa62943159`
- License: Apache License 2.0 (see `LICENSE` in this folder)
- Authors: Yin Chen, Jia Li, Shiguang Shan, Meng Wang, Richang Hong
- Paper: Chen et al., "From Static to Dynamic: Adapting Landmark-Aware Image Models for Facial
  Expression Recognition in Videos", IEEE Transactions on Affective Computing, 2024.

Files copied **unmodified**:
- `model_finetuning.py` (registers `s2d_base_patch16_224` with timm 0.6.13)
- `landmark/__init__.py`, `landmark/backbone.py`, `landmark/basenet.py`, `landmark/mobilefacenet.py`,
  `landmark/pfld_compressed.py` (MobileFaceNet landmark branch)

Not committed: `landmark/mobilefacenet_model_best.pth.tar` (12 MB). `00_setup.ipynb` downloads it
from the same commit into this folder, where `landmark/backbone.py` expects it (git-ignored).
