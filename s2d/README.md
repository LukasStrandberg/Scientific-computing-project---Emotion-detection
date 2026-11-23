# S2D track

**Method.** S2D ("From Static to Dynamic") adapts a ViT-B/16 trained on static facial expressions
(AffectNet) to video. It adds landmark-aware prompts: Multi-view Complementary Prompters (MCP), fed
by a frozen MobileFaceNet facial-landmark network, and Temporal-Modeling Adapters (TMA). Only these
and the head are trained, about 9.3 M of 95 M parameters. We start from the released model trained
on MAFW (11 classes) and adapt it to ElderReact's 6 emotions.

> Y. Chen, J. Li, S. Shan, M. Wang, R. Hong. *From Static to Dynamic: Adapting Landmark-Aware Image
> Models for Facial Expression Recognition in Videos.* IEEE Transactions on Affective Computing, 2024
> (arXiv:2312.05447).

**Upstream code and license.** github.com/MSA-LMC/S2D at commit `e1b5e5d`, Apache-2.0. The files
used (`model_finetuning.py`, `landmark/*.py`) are vendored **unmodified** in `third_party/s2d/`
(see `NOTICE.md` there). They run on timm 0.6.13, the version S2D was written for.

## Weights
| File | Source | Path (under the data root) |
|---|---|---|
| MAFW fold 1, without oversampler | huggingface.co/cyinen/S2D → `MAFW/without_oversampler/split01_checkpoint-best.pth` | `checkpoints/s2d/mafw_split01.pth` |
| MobileFaceNet landmarks | S2D repo (auto-download by `00_setup`) | `third_party/s2d/landmark/mobilefacenet_model_best.pth.tar` (repo) |
| AffectNet-7 pretrain (not used by default) | huggingface.co/cyinen/S2D → `pretrain-AffectNet-7.pth` | anywhere; only for your own experiments |

## Runs
| Run | What it is |
|---|---|
| `baseline` | the released MAFW model, zero-shot. Scores are the softmax probabilities of anger, disgust, fear, happiness, sadness and surprise among MAFW's 11 classes (rows 0, 4, 5, 6, 9, 10). F1@0.5 is not meaningful for these scores; AUC, AP and tuned F1 are. |
| `finetuned` | 6-output sigmoid head initialised from those MAFW rows; MCP + TMA + head trained on ElderReact |

## Preprocessing (shared, see the root README) and deviations from upstream
- Input: 16 frames, 224×224, ImageNet mean/std, the same as upstream.
- Crops: RetinaFace-R50 (the detector family the S2D authors used through deepface), largest face, eyes levelled, **deepface-style** crop (tight box, black padding). This beat a square crop on MAFW fold 1 with the released model (300 clips: 42.67 vs 40.33 WAR).
- **Deviation:** upstream tests on MAFW with two windows of 16 *consecutive* frames. Here every clip is sampled *uniformly* (16 bins over the whole clip; evaluation averages two views at 1/4 and 3/4 of each bin). On MAFW fold 1 the uniform sampling scored higher with the released model: 48.18 WAR / 33.59 UAR vs 44.21 / 30.39.
- **Deviation:** the crop cache holds 32 frames per clip instead of every frame. S2D's training jitter (one random frame per bin) becomes one random frame of each pair.

## Training settings (config section `s2d`, shared ones under `train`)
| | |
|---|---|
| trainable | `adapters`: parameters whose name contains `temporal_embedding, temporal_attn, cls_token_t, ln_post, cls_head, Adapter, head, prompt` (official S2D rule); `head`: classifier only |
| optimiser | AdamW, lr 5e-5, wd 0.05, betas (0.9, 0.999), 1 warm-up epoch + cosine to 0 |
| epochs | max 10 (cosine length), early stopping (patience 8) on validation loss |
| batch | 2 × gradient accumulation 8 = 16 (fits a 12 GB RTX 3060) |
| loss | BCE-with-logits, `pos_weight = clip(neg/pos, 1, 10)` |
| augmentation | horizontal flip p=0.5, temporal jitter; the landmark network stays frozen in eval mode |

Variants without notebook sections (give each its own `s2d.run` name): `--set s2d.trainable=head
--set s2d.lr=1e-3` (classifier only), `--set s2d.epochs=30` (longer run).

## Grad-CAM
Target: the token activations **entering the last ViT block** (`blocks[-1]`, forward pre-hook). S2D
classifies from the CLS token only, so the patch tokens at the last block's output get no gradient.
At its input they still reach the CLS token through attention. Channel weights are the gradient
averaged over all patch tokens of all 16 frames; CAM = ReLU(Σ w·A) on the 16×14×14 token grid,
upsampled to 16×224×224 and normalised per clip. `common/gradcam.py` holds the method;
`S2DTrack.cam_target` holds the layer and token reshape. `layer="patch_embed"` gives the
low-level variant.

## CLI
```bash
python tools/run.py preprocess --model s2d
python tools/run.py train      --model s2d
python tools/run.py evaluate   --model s2d
python tools/run.py patients   --model s2d [--preset maedfer]
# variant in its own run folder (results/s2d/finetuned_head/), then evaluate it the same way
python tools/run.py train    --model s2d --set s2d.run=finetuned_head --set s2d.trainable=head --set s2d.lr=1e-3
python tools/run.py evaluate --model s2d --set s2d.run=finetuned_head --runs finetuned_head
```

## Outputs
`results/s2d/{baseline,finetuned}/` in the shared schema (root README), plus `finetuned/best.pth`
(full state dict, ~380 MB) and `finetuned/log.csv`.

## Previous results (old protocol)
These numbers are from the `s2d` branch and are **not** comparable with the unified protocol. They
used the official train/dev split (no pooling), best epoch by dev AUC without early stopping,
thresholds tuned on the official dev set, RetinaFace deepface crops of every frame, and uniform
2-window sampling. ElderReact test, 353 clips. mAP and F1@0.5 were recomputed from the saved
`predictions.csv`.

| Run | mean AUC [95 % CI] | mAP | mean F1 (dev thresholds) [95 % CI] | F1@0.5 |
|---|---|---|---|---|
| MAFW zero-shot | 63.69 [60.16, 67.17] | 37.6 | 40.81 [37.02, 44.60] | 7.4 (softmax scores) |
| fine-tuned, 10 epochs, lr 5e-5 (chosen) | 66.44 [62.82, 69.85] | 39.1 | 43.05 [39.54, 46.61] | 42.5 |
| fine-tuned, 30 epochs, lr 1e-4 | 64.97 [60.93, 68.73] | 39.9 | 43.09 [38.98, 46.92] | 41.7 |
| classifier only, 10 epochs, lr 1e-3 | 65.41 [61.95, 68.73] | 38.7 | 43.21 [39.63, 46.52] | 37.8 |

Paired bootstrap, 10-epoch run vs zero-shot: AUC +2.76 [+0.01, +5.25]. Per emotion, sadness improved
by +9.1 [+3.1, +15.0] and happiness got worse by −5.3 [−9.0, −1.8].
t-SNE silhouette (cosine, single-emotion test clips): −0.118 before → −0.054 after fine-tuning (30-epoch model).
MAFW fold 1, released model: 48.18 WAR / 33.59 UAR (uniform), 44.21 / 30.39 (consecutive); the paper reports about 56 WAR with the authors' own crops.
