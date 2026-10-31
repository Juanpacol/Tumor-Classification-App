"""Quality gate and the two model inputs: radiomics image and CNN tensor."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.contracts import QualityMetrics
from src.preprocessing.brain_mask import foreground_mask, region_of_interest
from src.preprocessing.quality import QualityThresholds, check_quality, measure_quality

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing options. ``min_mask_fraction`` is the smallest share of the
    image the brain mask may cover before the full image is used instead.
    """

    min_mask_fraction: float = 0.05
    cnn_size: int = 224
    cnn_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    cnn_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)


@dataclass(frozen=True)
class PreprocessedImage:
    """``radiomics_image`` is the loaded (H, W) float32 image in [0, 1]; ``mask`` is its
    (H, W) boolean region of interest (the brain mask, or the full image when the mask
    is too small); ``cnn_input`` is (3, S, S) float32.
    """

    radiomics_image: np.ndarray
    mask: np.ndarray
    cnn_input: np.ndarray
    quality: QualityMetrics


def prepare_image(
    gray: np.ndarray, config: PreprocessingConfig | None = None
) -> PreprocessedImage:
    """Run the quality gate on a [0, 1] image and build both model inputs.

    Quality is measured inside the brain mask before its full-image fallback,
    so the fallback cannot raise the quality of a bad scan. Raises
    ``QualityCheckError`` when it is below the thresholds.
    """
    config = config or PreprocessingConfig()
    brain = foreground_mask(gray)
    quality = measure_quality(gray, mask=brain)
    check_quality(quality, config.thresholds)

    mask = region_of_interest(brain, config.min_mask_fraction)
    if mask is not brain:
        logger.warning(
            "brain mask covers %.1f%% of the image, under %.1f%%: using the full image",
            100 * brain.mean(),
            100 * config.min_mask_fraction,
        )
    return PreprocessedImage(gray, mask, _to_cnn_input(gray, config), quality)


def _to_cnn_input(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    size = config.cnn_size
    interpolation = cv2.INTER_AREA if min(image.shape) > size else cv2.INTER_LINEAR
    resized = cv2.resize(image, (size, size), interpolation=interpolation)
    channels = np.repeat(resized[None], 3, axis=0)
    mean = np.asarray(config.cnn_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(config.cnn_std, dtype=np.float32)[:, None, None]
    return ((channels - mean) / std).astype(np.float32)
