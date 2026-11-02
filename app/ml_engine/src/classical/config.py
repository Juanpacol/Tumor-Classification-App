"""Settings of the radiomic extractors.

Radiomic values depend on these parameters, so keep them explicit and store
them next to any model trained on the features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pywt


@dataclass(frozen=True)
class RadiomicsConfig:
    gray_levels: int = 32
    glcm_distances: tuple[int, ...] = (1,)
    glcm_angles: tuple[float, ...] = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)
    lbp_points: int = 8
    lbp_radius: int = 1
    wavelet: str = "haar"
    wavelet_level: int = 2
    percentiles: tuple[int, ...] = (10, 25, 50, 75, 90)
    min_mask_pixels: int = 16

    def __post_init__(self) -> None:
        positive = {
            "glcm_distances": self.glcm_distances,
            "lbp_points": (self.lbp_points,),
            "lbp_radius": (self.lbp_radius,),
            "wavelet_level": (self.wavelet_level,),
            "min_mask_pixels": (self.min_mask_pixels,),
        }
        for name, values in positive.items():
            if not values or min(values) < 1:
                raise ValueError(f"{name} must be at least 1, got {values}")
        if self.gray_levels < 2:
            raise ValueError(f"gray_levels must be at least 2, got {self.gray_levels}")
        if not self.glcm_angles:
            raise ValueError("glcm_angles must not be empty")
        if not self.percentiles or not all(0 <= p <= 100 for p in self.percentiles):
            raise ValueError(f"percentiles must be in [0, 100], got {self.percentiles}")
        if self.wavelet not in pywt.wavelist(kind="discrete"):
            raise ValueError(f"unknown discrete wavelet: {self.wavelet}")
