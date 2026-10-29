"""Uniform local binary pattern histogram over the mask."""

from __future__ import annotations

import math

import numpy as np
from skimage.feature import local_binary_pattern

from src.classical.config import RadiomicsConfig
from src.classical.roi import crop_to_mask


def feature_names(config: RadiomicsConfig) -> list[str]:
    return [f"lbp_uniform_{i}" for i in range(config.lbp_points + 2)]


def extract_lbp(
    image: np.ndarray, mask: np.ndarray, config: RadiomicsConfig
) -> dict[str, float]:
    """Normalized histogram of uniform LBP codes (``points + 2`` bins).

    The image is converted to 8 bits first because scikit-image's LBP is
    defined for integer images. Only masked pixels at least ``lbp_radius`` away
    from the edge of the region's bounding box are counted, because the edge
    pixels would be compared with padding. A too small region gives zeros.
    """
    names = feature_names(config)
    cropped = crop_to_mask(image, mask, config.min_mask_pixels)
    if cropped is None:
        return dict.fromkeys(names, 0.0)

    window, window_mask = cropped
    image8 = np.round(np.clip(window, 0.0, 1.0) * 255).astype(np.uint8)
    codes = local_binary_pattern(
        image8, config.lbp_points, config.lbp_radius, method="uniform"
    )
    margin = math.ceil(config.lbp_radius)
    interior = np.zeros(window_mask.shape, dtype=bool)
    interior[margin:-margin, margin:-margin] = True
    counted = window_mask & interior
    if not counted.any():
        return dict.fromkeys(names, 0.0)
    histogram = np.bincount(codes[counted].astype(int), minlength=len(names))
    return dict(zip(names, (histogram / histogram.sum()).astype(float)))
