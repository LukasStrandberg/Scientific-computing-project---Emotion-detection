# MAE-DFER track

**Method.** MAE-DFER pre-trains a video transformer as a masked autoencoder on VoxCeleb2 face videos
(self-supervised; it reconstructs masked appearance and frame differences). The encoder is an
efficient LGI-Former: 16 blocks, 512-d, local attention inside 2×5×10 token regions plus global
attention between per-region tokens. It is then fine-tuned for dynamic facial expression recognition
on DFEW (7 classes). We start from the authors' DFEW-fine-tuned model and train a new 6-emotion head
on ElderReact.

> L. Sun, Z. Lian, B. Liu, J. Tao. *MAE-DFER: Efficient Masked Autoencoder for Self-supervised
> Dynamic Facial Expression Recognition.* ACM Multimedia 2023, pp. 6110–6121.

**Upstream code and license.** github.com/sunlicai/MAE-DFER at commit `81fcf58`, MIT. Only
`modeling_finetune.py` is used; it is vendored **unmodified** in `third_party/mae_dfer/`.

**timm.** Upstream pins timm 0.4.12; this repository uses timm 0.6.13 (S2D needs it). The only
incompatibility is that `timm.create_model` in 0.6.x passes `pretrained_cfg=None` to the model
function, which upstream's `VisionTransformer` rejects. `mae_dfer/model.py` therefore calls the
registered function `vit_base_dim512_no_depth_patch16_160(...)` directly, and no file is patched.
The model uses only `drop_path`, `to_2tuple`, `trunc_normal_` and `register_model` from timm, which
are unchanged in 0.6.13, so the module tree and state-dict keys (519 tensors, 84,878,343
parameters) are identical. `tests/test_models.py` checks this against a key list generated from
the upstream repository.

## Weights
| File | Source | Path (under the data root) |
|---|---|---|
| DFEW fine-tuned, fold 1 (the track's `checkpoint-99.pth`) | MAE-DFER README → DFEW table → fold 1 "checkpoint" (Google Drive) | `checkpoints/mae_dfer/dfew_fold1.pth` |
| VoxCeleb2 pretrained (only for `maedfer.init: pretrained`) | MAE-DFER README → "pre-trained model on VoxCeleb2" | `checkpoints/mae_dfer/voxceleb2_checkpoint-49.pth` |
| YOLOv8n-face (only for `faces.detector: yolo`) | e.g. github.com/akanametov/yolo-face releases | `checkpoints/yolov8n-face.pt` |

Any DFEW fold works: none of them saw ElderReact. `config.json` of every run records the SHA-256 of
the file actually used. The previous fine-tuned weights (`MAE-DFER-ElderReact-yolo.pth`) are not
needed; `02_train` re-creates the run under the unified protocol.

## Runs
| Run | What it is |
|---|---|
| `baseline` | the DFEW model, zero-shot. Scores are the softmax probabilities of DFEW classes angry, disgust, fear, happy, sad, surprise (rows 3, 5, 6, 0, 1, 4; DFEW order happy, sad, neutral, angry, surprise, disgust, fear). F1@0.5 is not meaningful for these scores. |
| `finetuned` | 6-output sigmoid head initialised from those DFEW rows. Default `freeze: all` trains head + `fc_norm` only (linear probe), the previous best configuration of this track. |

## Preprocessing (shared) and deviations from the previous MAE-DFER track
- Input: 16 frames, **160×160** (resized from the shared 224 crops, bilinear with antialiasing), ImageNet mean/std, the same as upstream (`datasets.py`).
- **Deviation (detector):** the previous track used YOLOv8n-face with **one box per clip** (detected on the middle / quarter frames), a square crop with a 5 % margin, and no alignment. The shared protocol uses RetinaFace **per frame**, eye-levelled, deepface-style crop. Reason: MAE-DFER was fine-tuned on DFEW's per-frame face-region crops (and pre-trained on VoxCeleb2 face tracks), not on a fixed clip box. The old protocol stays available: `--set faces.detector=yolo --set faces.per_clip_box=true --set faces.style=square --set faces.margin=0.05`. `01_preprocess` with `RUN_DIAGNOSTIC = True` compares both detectors on your data.
- **Deviation (evaluation):** the previous track evaluated one 16-frame view (even frames of 32). Here two views (even and odd) are averaged, the same as for S2D. Upstream DFEW testing also averages 2 temporal segments.
- Upstream DFEW training samples 16 frames with stride 4; ElderReact clips are 1–5 s, so uniform sampling over the clip is used instead (as before).

## Training settings (config section `maedfer`, shared ones under `train`)
| | |
|---|---|
| freeze | `all` (head + fc_norm, 4.1 k parameters; frozen blocks run in eval mode, no drop-path) / `partial` (+ last `unfreeze_last_blocks` blocks) / `none` |
| optimiser | AdamW, head lr 5e-4, backbone lr 5e-5 (only if unfrozen), wd 0.05, **1** warm-up epoch (previously 3) + cosine |
| epochs | max 30, early stopping (patience 8) on validation loss |
| batch | 8, drop-path 0.1 only when blocks are trainable |
| loss / augmentation | BCE-with-logits with `pos_weight = clip(neg/pos, 1, 10)`; hflip p=0.5, temporal jitter |

Variants without notebook sections (give each its own `maedfer.run` name): `--set maedfer.freeze=partial`, `--set maedfer.freeze=none`,
`--set maedfer.init=pretrained`, and `--set eval.decision_rules=true` (the old decision-rule table:
thresholds / top-k, micro / macro / sample F1, stored in `metrics.json`).

## Patient-video feature re-centring (optional)
The previous inference standardised each patient video's 512-d features and mapped them onto the
mean / std of the ElderReact features before applying the head again (a test-time domain
adaptation). It is available for both supervised tracks as `patient.feature_recentering: true`,
**off by default**: it changes the predictions and has no counterpart for clustering. The source
statistics come from the run's ElderReact val + test features.

## Grad-CAM
Target: the tokens **entering the last LGI-Former block**. The classifier uses only the 8 region
tokens, so original tokens at the block output get no gradient; at its input they reach the region
tokens through local attention. Tokens are regrouped from the 8 regions to the 8×10×10 grid (2-frame
tubelets × 10×10 patches); channel weights are mean gradients; CAM = ReLU(Σ w·A), upsampled to
16×160×160 → shown on the 224 crops. The previous track hooked `patch_embed.proj` (first layer);
that variant is `layer="patch_embed"`.

## CLI
```bash
python tools/run.py preprocess --model maedfer
python tools/run.py train      --model maedfer
python tools/run.py evaluate   --model maedfer
python tools/run.py patients   --model maedfer --preset maedfer   # the track's original windows
```

## Outputs
`results/maedfer/{baseline,finetuned}/` in the shared schema, plus `finetuned/best.pth` and `log.csv`.

## Previous results (old protocol)
These are from `Isaks_branch` (`MAE-DFER.ipynb`, YOLO run). Same pooled split as now, but YOLO
per-clip square crops (5 % margin, 160 px), a single evaluation view, warm-up 3 epochs, linear probe
(`freeze=all`, `LR_HEAD=5e-4`). Best epoch 6 of 14 (early stop) by validation loss. Not comparable
with the unified protocol.

| Split | loss | mean F1@0.5 | mAP |
|---|---|---|---|
| validation (146 clips) | 0.833 | 0.529 | 0.516 |
| test (353 clips) | 0.862 | **0.430** | **0.406** |

Per-emotion test F1@0.5: anger 0.383, disgust 0.389, fear 0.268, happiness 0.735, sadness 0.353,
surprise 0.454. Zero-shot DFEW model on validation: mAP 0.416 with the class mapping (base rate
0.307). The best-matching DFEW column was "surprise" for fear and "sad" for surprise, so the mapping
for those two is weak.
