"""Quality gate and the two model inputs: radiomics image and CNN tensor."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.contracts import QualityMetrics
from src.preprocessing.quality import QualityThresholds, check_quality, measure_quality
from src.preprocessing.skull_stripping import foreground_mask, strip_skull


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing options. Skull stripping is off by default: the literature
    review (docs/research) treats it as an ablation variable, not a default.
    """

    skull_stripping: bool = False
    cnn_size: int = 224
    cnn_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    cnn_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)


@dataclass(frozen=True)
class PreprocessedImage:
    """``radiomics_image`` is (H, W) float32 in [0, 1]; ``mask`` is its (H, W) boolean region
    of interest (the brain mask, or the foreground when skull stripping was not applied);
    ``cnn_input`` is (3, S, S) float32.
    """

    radiomics_image: np.ndarray
    mask: np.ndarray
    cnn_input: np.ndarray
    quality: QualityMetrics


def prepare_image(
    gray: np.ndarray, config: PreprocessingConfig | None = None
) -> PreprocessedImage:
    """Run the quality gate on a [0, 1] image and build both model inputs.

    Quality is measured on the original image, so skull stripping cannot
    hide a bad scan. Raises ``QualityCheckError`` when it is below the
    thresholds.
    """
    config = config or PreprocessingConfig()
    stripped = strip_skull(gray) if config.skull_stripping else None
    applied = bool(stripped and stripped.applied)
    foreground = foreground_mask(gray)
    quality = measure_quality(gray, skull_stripped=applied, mask=foreground)
    check_quality(quality, config.thresholds)

    image = stripped.image if stripped else gray
    mask = stripped.mask if applied else foreground
    return PreprocessedImage(image, mask, _to_cnn_input(image, config), quality)


def _to_cnn_input(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    size = config.cnn_size
    interpolation = cv2.INTER_AREA if min(image.shape) > size else cv2.INTER_LINEAR
    resized = cv2.resize(image, (size, size), interpolation=interpolation)
    channels = np.repeat(resized[None], 3, axis=0)
    mean = np.asarray(config.cnn_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(config.cnn_std, dtype=np.float32)[:, None, None]
    return ((channels - mean) / std).astype(np.float32)
