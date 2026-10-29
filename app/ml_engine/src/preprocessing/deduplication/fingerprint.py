"""Per-image fingerprint used to compare images against each other."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing.cropping import crop_to_foreground
from src.preprocessing.deduplication.hashing import (
    normalized_thumbnail,
    perceptual_hash,
    to_gray_uint8,
)

_AUGMENTED_PATTERN = re.compile(r"(^|[-_])aug([-_]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class ImageFingerprint:
    """Hashes and thumbnails of one image, plus the metadata needed to pick a representative."""

    path: Path
    split: str
    label: str
    area: int
    is_augmented: bool
    pixel_sha256: str
    phash: int
    phash_flipped: int
    phash_cropped: int
    phash_cropped_flipped: int
    thumbnail: np.ndarray = field(compare=False, repr=False)
    thumbnail_cropped: np.ndarray = field(compare=False, repr=False)


def is_augmented_name(path: Path) -> bool:
    """Whether the file name marks a synthetic augmentation (e.g. ``Tr-aug-me_3``)."""
    return bool(_AUGMENTED_PATTERN.search(path.stem))


def compute_fingerprint(path: Path, split: str, label: str) -> ImageFingerprint:
    """Hash the full and background-cropped versions of an image, direct and mirrored."""
    with Image.open(path) as image:
        gray = to_gray_uint8(image)

    pixel_digest = hashlib.sha256(np.asarray(gray.shape, dtype=np.int64).tobytes())
    pixel_digest.update(gray.tobytes())

    full = Image.fromarray(gray)
    cropped = Image.fromarray(crop_to_foreground(gray))
    mirror = Image.Transpose.FLIP_LEFT_RIGHT
    return ImageFingerprint(
        path=path,
        split=split,
        label=label,
        area=gray.size,
        is_augmented=is_augmented_name(path),
        pixel_sha256=pixel_digest.hexdigest(),
        phash=perceptual_hash(full),
        phash_flipped=perceptual_hash(full.transpose(mirror)),
        phash_cropped=perceptual_hash(cropped),
        phash_cropped_flipped=perceptual_hash(cropped.transpose(mirror)),
        thumbnail=normalized_thumbnail(full),
        thumbnail_cropped=normalized_thumbnail(cropped),
    )


def thumbnail_correlation(first: ImageFingerprint, second: ImageFingerprint) -> float:
    """Highest Pearson correlation over full/cropped thumbnails, direct and mirrored."""
    return max(
        float(np.mean(a * b))
        for a_full, b_full in (
            (first.thumbnail, second.thumbnail),
            (first.thumbnail_cropped, second.thumbnail_cropped),
        )
        for a, b in ((a_full, b_full), (a_full[:, ::-1], b_full))
    )
