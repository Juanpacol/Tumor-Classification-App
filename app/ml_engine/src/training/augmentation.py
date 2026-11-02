"""Train-only augmented copies of the CNN image (pipeline block B7)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentationConfig:
    """Number of copies per usable Training image and the range of each transform."""

    copies: int = 1
    max_rotation_deg: float = 10.0
    flip_probability: float = 0.5
    scale_range: float = 0.1
    brightness_range: float = 0.1
    contrast_range: float = 0.1

    def __post_init__(self) -> None:
        if self.copies < 0:
            raise ValueError(f"copies must be at least 0, got {self.copies}")
        if not 0 <= self.max_rotation_deg <= 180:
            raise ValueError(
                f"max_rotation_deg must be in [0, 180], got {self.max_rotation_deg}"
            )
        if not 0 <= self.flip_probability <= 1:
            raise ValueError(
                f"flip_probability must be in [0, 1], got {self.flip_probability}"
            )
        ranges = {
            "scale_range": self.scale_range,
            "brightness_range": self.brightness_range,
            "contrast_range": self.contrast_range,
        }
        for name, value in ranges.items():
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1), got {value}")


@dataclass(frozen=True)
class AugmentationParams:
    """The transform of one copy."""

    rotation_deg: float
    flip: bool
    scale: float
    brightness: float
    contrast: float


def copy_rng(seed: int, sha256: str, index: int) -> np.random.Generator:
    """Generator of one copy, from the run seed, the original's content hash and the
    copy index, so a copy does not depend on the order images are processed in."""
    return np.random.default_rng([seed, int(sha256[:16], 16), index])


def draw(config: AugmentationConfig, rng: np.random.Generator) -> AugmentationParams:
    """Parameters of one copy, each uniform in its range."""
    return AugmentationParams(
        rotation_deg=float(
            rng.uniform(-config.max_rotation_deg, config.max_rotation_deg)
        ),
        flip=bool(rng.random() < config.flip_probability),
        scale=float(rng.uniform(1 - config.scale_range, 1 + config.scale_range)),
        brightness=float(
            rng.uniform(1 - config.brightness_range, 1 + config.brightness_range)
        ),
        contrast=float(
            rng.uniform(1 - config.contrast_range, 1 + config.contrast_range)
        ),
    )


def keeps_augmentation(
    mean_without: float, std_without: float, mean_with: float
) -> bool:
    """Whether augmentation stays on: its mean CV macro-F1 must beat the run without
    copies by at least that run's standard deviation across folds, so fold noise
    alone cannot turn it on."""
    return mean_with >= mean_without + std_without


def augment(image: np.ndarray, params: AugmentationParams) -> np.ndarray:
    """Copy of an ``(S, S)`` CNN image in [0, 1], same shape, black beyond the border.

    Flip, then rotation and scale in one warp (one interpolation), then contrast
    around the image mean and a multiplicative brightness. Black pixels (the warp
    border and the padding of the geometry step) stay black, so a copy never gets a
    background that no original has.
    """
    flipped = np.fliplr(image) if params.flip else image
    height, width = flipped.shape
    matrix = cv2.getRotationMatrix2D(
        ((width - 1) / 2, (height - 1) / 2), params.rotation_deg, params.scale
    )
    warped = cv2.warpAffine(
        np.ascontiguousarray(flipped, dtype=np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float64)
    mean = warped.mean()
    adjusted = params.brightness * (params.contrast * (warped - mean) + mean)
    adjusted[warped == 0.0] = 0.0
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32)
