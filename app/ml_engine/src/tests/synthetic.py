"""Synthetic MRI-like images for tests."""

from __future__ import annotations

import csv
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


TEXTURE_SIGMAS = {"glioma": 3.0, "meningioma": 6.0, "notumor": 12.0, "pituitary": 24.0}


def write_labeled_dataset(root: Path, per_class: int) -> tuple[Path, Path]:
    """Four texture classes in Train/ and Test/ folders plus the cleaning ``manifest.csv``."""
    data_dir = root / "data"
    rows = []
    group = 0
    for split in ("Train", "Test"):
        for label, sigma in TEXTURE_SIGMAS.items():
            for index in range(per_class):
                relative = f"{split}/{label}/{index}.png"
                save(textured_slice(group, sigma), data_dir / relative)
                rows.append([relative, split, label, group, 1, 1, "kept"])
                group += 1
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["path", "split", "label", "group_id", "group_size", "keep", "reason"]
        )
        writer.writerows(rows)
    return data_dir, manifest


def separable_features(
    n_per_class: int = 30,
    seed: int = 0,
    n_radiomic: int = 45,
    n_cnn: int = 20,
    informative: tuple[str, ...] = ("radiomic", "cnn"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature matrix, labels and groups for four classes that are separable only in the
    ``informative`` blocks; the other block is pure noise. Groups pair near-duplicates.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(list(TEXTURE_SIGMAS))[np.repeat(np.arange(4), n_per_class)]
    blocks = []
    for name, width in (("radiomic", n_radiomic), ("cnn", n_cnn)):
        noise = rng.normal(size=(len(labels), width))
        if name in informative:
            centers = rng.normal(scale=3.0, size=(4, width))
            noise += centers[np.repeat(np.arange(4), n_per_class)]
        blocks.append(noise)
    groups = np.arange(len(labels)) // 2
    return np.hstack(blocks), labels, groups
