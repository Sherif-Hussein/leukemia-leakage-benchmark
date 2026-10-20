"""Stain augmentation: random perturbation in haematoxylin-eosin-DAB optical-density space
(Tellez et al., 2018), which simulates staining differences between laboratories."""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import hed2rgb, rgb2hed


class HEDJitter:
    def __init__(self, sigma: float = 0.05, bias: float = 0.02, p: float = 0.8):
        self.sigma, self.bias, self.p = sigma, bias, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        hed = rgb2hed(np.asarray(img.convert("RGB")))
        alpha = np.random.uniform(1 - self.sigma, 1 + self.sigma, 3)
        beta = np.random.uniform(-self.bias, self.bias, 3)
        rgb = np.clip(hed2rgb(hed * alpha + beta), 0, 1)
        return Image.fromarray((rgb * 255).astype(np.uint8))
