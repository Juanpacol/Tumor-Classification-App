"""Wavelet sub-band energy, mean and std over the mask."""

from __future__ import annotations

import cv2
import numpy as np
import pywt

from src.classical.config import RadiomicsConfig
from src.classical.roi import crop_to_mask

_STATISTICS = ("energy", "mean", "std")


def _band_names(config: RadiomicsConfig) -> list[str]:
    level = config.wavelet_level
    names = [f"cA{level}"]
    for current in range(level, 0, -1):
        names += [f"cH{current}", f"cV{current}", f"cD{current}"]
    return names


def feature_names(config: RadiomicsConfig) -> list[str]:
    return [
        f"wavelet_{band}_{stat}" for band in _band_names(config) for stat in _STATISTICS
    ]


def extract_wavelet(
    image: np.ndarray, mask: np.ndarray, config: RadiomicsConfig
) -> dict[str, float]:
    """Energy (mean square), mean and std of every sub-band of a 2D DWT.

    The transform runs on the mask bounding box with the outside filled by the
    ROI mean, to avoid artificial edges. Statistics use only the sub-band
    positions that fall inside the mask, downsampled to each sub-band size.
    A region too small for the requested level gives zeros.
    """
    names = feature_names(config)
    cropped = crop_to_mask(image, mask, config.min_mask_pixels)
    if cropped is None:
        return dict.fromkeys(names, 0.0)

    window, window_mask = cropped
    max_level = pywt.dwt_max_level(
        min(window.shape), pywt.Wavelet(config.wavelet).dec_len
    )
    if config.wavelet_level > max_level:
        return dict.fromkeys(names, 0.0)
    filled = np.where(window_mask, window, window[window_mask].mean()).astype(
        np.float64
    )
    coefficients = pywt.wavedec2(filled, config.wavelet, level=config.wavelet_level)
    bands = [
        coefficients[0],
        *(band for details in coefficients[1:] for band in details),
    ]
    mask8 = window_mask.astype(np.uint8)

    features: dict[str, float] = {}
    for band_name, band in zip(_band_names(config), bands):
        height, width = band.shape
        band_mask = cv2.resize(mask8, (width, height), interpolation=cv2.INTER_NEAREST)
        values = band[band_mask.astype(bool)]
        stats = (
            (float(np.mean(values**2)), float(values.mean()), float(values.std()))
            if values.size
            else (0.0, 0.0, 0.0)
        )
        for stat_name, value in zip(_STATISTICS, stats):
            features[f"wavelet_{band_name}_{stat_name}"] = float(np.nan_to_num(value))
    return features
