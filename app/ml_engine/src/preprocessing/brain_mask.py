"""Lightweight 2D skull stripping: Otsu threshold plus morphology, CPU only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import disk

MIN_MASK_FRACTION = 0.15
MAX_MASK_FRACTION = 0.95
_RADIUS_FRACTION = 0.02


@dataclass(frozen=True)
class SkullStrippingResult:
    """Brain mask and masked image; ``applied`` is False when the fallback was used."""

    mask: np.ndarray
    image: np.ndarray
    applied: bool


def strip_skull(gray: np.ndarray) -> SkullStrippingResult:
    """Remove non-brain tissue from a 2D [0, 1] image.

    When the mask covers less than ``MIN_MASK_FRACTION`` or more than
    ``MAX_MASK_FRACTION`` of the image (for example an already cropped
    image) the input is returned unchanged with ``applied=False``. This
    function never raises.
    """
    mask = _brain_mask(gray)
    if mask is None or not MIN_MASK_FRACTION <= mask.mean() <= MAX_MASK_FRACTION:
        return SkullStrippingResult(np.ones(gray.shape, dtype=bool), gray, False)
    return SkullStrippingResult(
        mask, np.where(mask, gray, 0.0).astype(np.float32), True
    )


def _brain_mask(gray: np.ndarray) -> np.ndarray | None:
    """Largest bright component after eroding away the thin scalp and skull links."""
    if gray.max() <= gray.min():
        return None
    head = gray > threshold_otsu(gray)
    radius = max(1, round(min(gray.shape) * _RADIUS_FRACTION))
    head = ndimage.binary_opening(head, structure=disk(radius))
    core = ndimage.binary_erosion(head, structure=disk(2 * radius))
    labels, count = ndimage.label(core)
    if count == 0:
        return None
    largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    brain = ndimage.binary_dilation(labels == largest, structure=disk(2 * radius))
    return ndimage.binary_fill_holes(brain)


def bright_regions(gray: np.ndarray) -> np.ndarray:
    """Every pixel above half the Otsu threshold, all components kept. ``gray`` must not be flat."""
    return gray > 0.5 * threshold_otsu(gray)


def foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Pixels clearly above the background: half the Otsu threshold, largest component, holes filled.

    Used as the region of interest when skull stripping was not applied. A flat
    image is all foreground if it is not black, and empty otherwise.
    """
    if gray.max() <= gray.min():
        return np.full(gray.shape, gray.max() > 0, dtype=bool)
    foreground = bright_regions(gray)
    labels, count = ndimage.label(foreground)
    if count == 0:
        return foreground
    largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return ndimage.binary_fill_holes(labels == largest)
