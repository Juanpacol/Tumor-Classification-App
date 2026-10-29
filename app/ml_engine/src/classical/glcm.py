"""Gray-level co-occurrence matrix features over the mask."""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from src.classical.config import RadiomicsConfig
from src.classical.roi import crop_to_mask, quantize_roi

_PROPERTIES = ("contrast", "energy", "homogeneity", "correlation", "dissimilarity")


def feature_names(config: RadiomicsConfig) -> list[str]:
    return [f"glcm_{prop}" for prop in _PROPERTIES]


def extract_glcm(
    image: np.ndarray, mask: np.ndarray, config: RadiomicsConfig
) -> dict[str, float]:
    """GLCM contrast, energy, homogeneity, correlation and dissimilarity.

    The ROI is quantized to ``gray_levels`` and level 0 (background) is dropped
    from the matrix, so only pairs with both pixels inside the mask count. Each
    feature is computed per angle on a symmetric matrix and averaged over the
    angles that have pairs. ``energy`` is scikit-image's definition, the square
    root of the angular second moment. Flat or too small regions give zeros.
    """
    names = feature_names(config)
    cropped = crop_to_mask(image, mask, config.min_mask_pixels)
    quantized = quantize_roi(*cropped, config.gray_levels) if cropped else None
    if quantized is None:
        return dict.fromkeys(names, 0.0)

    glcm = graycomatrix(
        quantized,
        list(config.glcm_distances),
        list(config.glcm_angles),
        levels=config.gray_levels + 1,
        symmetric=True,
    )
    glcm = glcm[1:, 1:].astype(np.float64)
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    normalized = np.divide(glcm, totals, out=np.zeros_like(glcm), where=totals > 0)
    populated = totals[0, 0] > 0
    if not populated.any():
        return dict.fromkeys(names, 0.0)

    features = {}
    for prop, name in zip(_PROPERTIES, names):
        values = graycoprops(normalized, prop)[populated]
        features[name] = float(np.nan_to_num(values.mean()))
    return features
