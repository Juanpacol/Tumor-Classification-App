"""Quality gate and the two model inputs: radiomics image and CNN image."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.contracts import QualityMetrics
from src.preprocessing.brain_mask import foreground_mask, region_of_interest
from src.preprocessing.geometry import crop_square, resize_square, square_box
from src.preprocessing.quality import QualityThresholds, check_quality, measure_quality

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing options. ``min_mask_fraction`` is the smallest share of the
    image the brain mask may cover before the full image is used instead.
    ``crop_margin`` widens the CNN crop at each end by this fraction of the longer
    side of the region's bounding box.
    """

    min_mask_fraction: float = 0.05
    crop_margin: float = 0.05
    cnn_size: int = 224
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    def __post_init__(self) -> None:
        if not 0 <= self.crop_margin <= 1:
            raise ValueError(f"crop_margin must be in [0, 1], got {self.crop_margin}")


@dataclass(frozen=True)
class PreprocessedImage:
    """``radiomics_image`` is the loaded (H, W) float32 image in [0, 1]; ``mask`` is its
    (H, W) boolean region of interest (the brain mask, or the full image when the mask
    is too small); ``cnn_image`` is the (S, S) float32 CNN image in [0, 1], a square crop
    centered on that region (black beyond the image borders) resized to S. The CNN
    extractor turns it into the normalized tensor of its backbone.
    """

    radiomics_image: np.ndarray
    mask: np.ndarray
    cnn_image: np.ndarray
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
    square = crop_square(gray, square_box(mask, config.crop_margin))
    cnn_image = resize_square(square, config.cnn_size)
    return PreprocessedImage(gray, mask, cnn_image, quality)
