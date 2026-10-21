"""Frozen deep encoders -> one embedding per cell image.

Supported names
  effnet_b0_imagenet  timm efficientnet_b0 (ImageNet), 1280-d. The typical "transfer learning" baseline.
  dinobloom_s / _b / _l / _g   DinoBloom (Koch et al., MICCAI 2024), 384/768/1024/1536-d CLS token.
                      Weights: https://zenodo.org/records/10908163  (DinoBloom-S.pth etc.)
                      Loading follows the official code (marrlab/DinoBloom, dinov2/eval/models/return_model.py):
                      DINOv2 hub architecture, "teacher" weights with the "backbone." prefix removed, and a
                      257-token positional embedding (224 x 224 input, patch 14).
Both use ImageNet mean/std normalisation and 224 x 224 input, as in the original code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINOBLOOM_ARCH = {"dinobloom_s": ("dinov2_vits14", 384), "dinobloom_b": ("dinov2_vitb14", 768),
                  "dinobloom_l": ("dinov2_vitl14", 1024), "dinobloom_g": ("dinov2_vitg14", 1536)}


def build_encoder(name: str, weights: str | None = None) -> nn.Module:
    if name == "effnet_b0_imagenet":
        import timm
        kw = {"pretrained_cfg_overlay": dict(file=weights)} if weights else {}
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, **kw)
    elif name in DINOBLOOM_ARCH:
        if not weights:
            raise SystemExit(f"{name} needs --weights pointing to the DinoBloom .pth file from Zenodo")
        import timm
        arch = {"dinobloom_s": "vit_small_patch14_dinov2", "dinobloom_b": "vit_base_patch14_dinov2",
                "dinobloom_l": "vit_large_patch14_dinov2", "dinobloom_g": "vit_giant_patch14_dinov2"}[name]
        model = timm.create_model(arch, pretrained=False, img_size=224, num_classes=0)   # CLS token, 257 positions
        ckpt = torch.load(weights, map_location="cpu")
        sd = {k.replace("backbone.", ""): v for k, v in ckpt["teacher"].items() if k.startswith("backbone.") and "mask_token" not in k}
        model.load_state_dict(sd, strict=True)
    else:
        raise ValueError(f"unknown encoder {name}")
    return model.eval()


def default_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageList(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.t = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.t(Image.open(self.paths[i]).convert("RGB")), True
        except Exception:                      # unreadable file -> NaN embedding row
            return self.t(Image.new("RGB", (224, 224))), False


@torch.no_grad()
def embed(model: nn.Module, paths, device: str = "cpu", batch_size: int = 64, workers: int = 2,
          log_every: int = 50):
    import numpy as np
    model = model.to(device)
    dl = DataLoader(ImageList(paths, default_transform()), batch_size=batch_size, num_workers=workers)
    out = []
    for i, (x, ok) in enumerate(dl):
        z = model(x.to(device))
        if isinstance(z, (tuple, list)):
            z = z[0]
        z = z.float().cpu().numpy()
        z[~ok.numpy().astype(bool)] = np.nan
        out.append(z)
        if log_every and (i + 1) % log_every == 0:
            print(f"  {min((i + 1) * batch_size, len(paths)):,}/{len(paths):,}", flush=True)
    return np.concatenate(out).astype("float32")
