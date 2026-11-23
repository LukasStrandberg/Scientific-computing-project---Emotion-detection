# Vendored: MAE-DFER

- Source: https://github.com/sunlicai/MAE-DFER at commit `81fcf589bb584a7a255e47af0e802c12b25f9eba`
- License: MIT (see `LICENSE` in this folder), Copyright (c) 2023 Licai Sun
- Paper: Sun et al., "MAE-DFER: Efficient Masked Autoencoder for Self-supervised Dynamic Facial
  Expression Recognition", ACM Multimedia 2023.

File copied **unmodified**: `modeling_finetune.py` (LGI-Former / ViT definition).

timm compatibility: under timm 0.6.13, `timm.create_model` passes `pretrained_cfg=None` to the
model function, which `VisionTransformer.__init__` does not accept. `mae_dfer/model.py` therefore
calls the registered function `vit_base_dim512_no_depth_patch16_160(**kwargs)` directly instead of
going through `create_model`. No change to this file is needed.
