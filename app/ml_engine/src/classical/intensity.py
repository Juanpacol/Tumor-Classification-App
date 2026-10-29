"""First-order intensity statistics over the mask."""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.classical.config import RadiomicsConfig

_FLAT_TOLERANCE = 1e-12


def feature_names(config: RadiomicsConfig) -> list[str]:
    percentiles = [f"intensity_p{p}" for p in config.percentiles]
    return [
        "intensity_mean",
        "intensity_std",
        "intensity_skewness",
        "intensity_kurtosis",
        *percentiles,
    ]


def extract_intensity(
    image: np.ndarray, mask: np.ndarray, config: RadiomicsConfig
) -> dict[str, float]:
    """Mean, std, skewness, excess kurtosis and percentiles of the masked pixels.

    Excess kurtosis (IBSI convention). A flat or too small region gives zeros.
    """
    names = feature_names(config)
    values = image[mask].astype(np.float64)
    flat = values.size > 0 and np.ptp(values) <= _FLAT_TOLERANCE
    if values.size < config.min_mask_pixels or flat:
        features = dict.fromkeys(names, 0.0)
        if values.size >= config.min_mask_pixels:
            features["intensity_mean"] = float(values.mean())
            for p in config.percentiles:
                features[f"intensity_p{p}"] = float(np.percentile(values, p))
        return features

    computed = [
        values.mean(),
        values.std(),
        stats.skew(values),
        stats.kurtosis(values),
        *np.percentile(values, config.percentiles),
    ]
    return {name: float(np.nan_to_num(value)) for name, value in zip(names, computed)}
