"""Region-of-interest helpers shared by the extractors."""

from __future__ import annotations

import numpy as np


def crop_to_mask(
    image: np.ndarray, mask: np.ndarray, min_pixels: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Crop image and mask to the mask bounding box, or None if it has too few pixels."""
    if int(mask.sum()) < min_pixels:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    window = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    return image[window], mask[window]


def quantize_roi(image: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray | None:
    """Quantize ROI intensities to ``1..levels``; pixels outside the mask are 0.

    Level 0 is reserved for the background so it can be dropped from texture
    matrices. Returns None when the ROI is flat, where quantization is undefined.
    """
    roi = image[mask]
    low, high = float(roi.min()), float(roi.max())
    if high <= low:
        return None
    scaled = np.floor((image - low) / (high - low) * levels)
    quantized = np.clip(scaled, 0, levels - 1).astype(np.uint8) + 1
    return np.where(mask, quantized, 0).astype(np.uint8)
