"""Combined radiomic extractor with a fixed-length, ordered feature vector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.classical import glcm, intensity, lbp, wavelet
from src.classical.config import RadiomicsConfig

_FAMILIES = (
    (glcm.feature_names, glcm.extract_glcm),
    (lbp.feature_names, lbp.extract_lbp),
    (wavelet.feature_names, wavelet.extract_wavelet),
    (intensity.feature_names, intensity.extract_intensity),
)


@dataclass(frozen=True)
class RadiomicsFeatures:
    """``values`` is aligned with ``names``; both depend only on the config."""

    names: tuple[str, ...]
    values: np.ndarray


def feature_names(config: RadiomicsConfig | None = None) -> tuple[str, ...]:
    """Ordered feature names for a config, without computing anything."""
    config = config or RadiomicsConfig()
    return tuple(name for names, _ in _FAMILIES for name in names(config))


def extract_radiomics(
    image: np.ndarray, mask: np.ndarray, config: RadiomicsConfig | None = None
) -> RadiomicsFeatures:
    """Extract GLCM, LBP, wavelet and intensity features over ``mask``.

    The vector has the same length and order for any valid input and is always
    finite: flat or too small regions give zeros for the affected family.
    Raises ``ValueError`` when ``image`` is not 2D or ``mask`` is not a
    boolean array of the same shape.
    """
    config = config or RadiomicsConfig()
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    if mask.shape != image.shape or mask.dtype != bool:
        raise ValueError("mask must be a boolean array with the same shape as image")

    features: dict[str, float] = {}
    for _, extract in _FAMILIES:
        features.update(extract(image, mask, config))
    names = feature_names(config)
    return RadiomicsFeatures(names, np.array([features[name] for name in names]))
