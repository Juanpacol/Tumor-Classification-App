"""Settings of the radiomic extractors.

Radiomic values depend on these parameters, so keep them explicit and store
them next to any model trained on the features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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
