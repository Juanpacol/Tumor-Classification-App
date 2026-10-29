"""Cropping utilities for 2D brain MRI slices."""

from __future__ import annotations

import numpy as np


def crop_to_foreground(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Crop a 2D grayscale image to the bounding box of pixels brighter than ``threshold``.

    Returns the input unchanged when no pixel exceeds ``threshold``.
    """
    mask = image > threshold
    if not mask.any():
        return image
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
