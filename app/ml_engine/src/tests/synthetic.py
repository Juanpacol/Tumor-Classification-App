"""Synthetic MRI-like images for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def synthetic_slice(seed: int, size: int = 256) -> Image.Image:
    """Smooth random texture inside a disc on a black background."""
    rng = np.random.default_rng(seed)
    texture = gaussian_filter(rng.random((size, size)), sigma=12)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    yy, xx = np.mgrid[:size, :size]
    head = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size * 0.4) ** 2
    return Image.fromarray(np.where(head, 40 + texture * 215, 0).astype(np.uint8))


def save(image: Image.Image, path: Path, **kwargs: object) -> Path:
    """Save ``image`` creating parent folders, and return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, **kwargs)
    return path


def head_phantom(size: int = 256) -> np.ndarray:
    """[0, 1] float image of a bright brain inside a dark skull gap and a scalp ring."""
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(yy - size / 2, xx - size / 2) / size
    image = np.zeros((size, size), dtype=np.float32)
    image[radius < 0.45] = 0.5
    image[radius < 0.36] = 0.05
    image[radius < 0.32] = 0.7
    return image


def textured_slice(seed: int, sigma: float, size: int = 128) -> Image.Image:
    """Like ``synthetic_slice`` but with a chosen texture scale, to build separable classes."""
    rng = np.random.default_rng(seed)
    texture = gaussian_filter(rng.random((size, size)), sigma=sigma)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    yy, xx = np.mgrid[:size, :size]
    head = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size * 0.4) ** 2
    return Image.fromarray(np.where(head, 40 + texture * 215, 0).astype(np.uint8))
