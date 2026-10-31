"""Image quality measurements and the quality gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.restoration import estimate_sigma

from src.contracts import QualityMetrics
from src.errors import QualityCheckError
from src.preprocessing.brain_mask import bright_regions, foreground_mask

METRICS_VERSION = 2

_HISTOGRAM_BINS = 256
_NOISE_FLOOR = 1.0 / 255.0
_MAD_TO_SIGMA = 1.4826
_MIN_BACKGROUND_FRACTION = 0.01
_MARGIN_FRACTION = 0.02
_MIN_MARGIN = 3


@dataclass(frozen=True)
class QualityThresholds:
    """Minimum acceptable quality, for the metric definitions of ``metrics_version``.

    The defaults were calibrated on the purged training split on 2026-10-30
    and reviewed (docs/ml-engine/decisions.md, D17). ``metrics_version`` names
    the SNR and entropy definitions; it must change whenever they do.
    """

    min_snr: float = 2.74
    min_entropy: float = 3.2
    metrics_version: int = METRICS_VERSION


def shannon_entropy(gray: np.ndarray, mask: np.ndarray) -> float:
    """Shannon entropy in bits of the intensity histogram inside ``mask``."""
    counts, _ = np.histogram(gray[mask], bins=_HISTOGRAM_BINS, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def estimate_snr(gray: np.ndarray, mask: np.ndarray) -> float:
    """Median intensity inside ``mask`` divided by the noise of the dark background.

    The background is every pixel away from all bright regions, so tissue
    outside the brain is not read as noise. Noise is a MAD-based sigma floored
    at one 8-bit step; with almost no background the wavelet estimate of the
    whole image is used instead. A flat image returns 0.
    """
    if gray.max() <= gray.min() or not mask.any():
        return 0.0
    signal = float(np.median(gray[mask]))
    return signal / max(_noise_sigma(gray), _NOISE_FLOOR)


def _noise_sigma(gray: np.ndarray) -> float:
    margin = max(_MIN_MARGIN, round(_MARGIN_FRACTION * min(gray.shape)))
    background = ~ndimage.binary_dilation(bright_regions(gray), iterations=margin)
    if background.sum() < _MIN_BACKGROUND_FRACTION * gray.size:
        return float(estimate_sigma(gray))
    values = gray[background]
    return _MAD_TO_SIGMA * float(np.median(np.abs(values - np.median(values))))


def measure_quality(
    gray: np.ndarray, *, mask: np.ndarray | None = None
) -> QualityMetrics:
    """Measure SNR and entropy of a 2D [0, 1] image inside its brain mask.

    ``mask`` is the image's ``foreground_mask`` when the caller already has it.
    Skull stripping is not part of the pipeline, so ``skull_stripped`` is False.
    """
    if mask is None:
        mask = foreground_mask(gray)
    return QualityMetrics(
        snr=estimate_snr(gray, mask),
        entropy=shannon_entropy(gray, mask),
        skull_stripped=False,
    )


def check_quality(metrics: QualityMetrics, thresholds: QualityThresholds) -> None:
    """Raise ``QualityCheckError`` with the measured values when below a threshold."""
    failures = []
    if metrics.snr < thresholds.min_snr:
        failures.append(f"snr {metrics.snr:.2f} < {thresholds.min_snr}")
    if metrics.entropy < thresholds.min_entropy:
        failures.append(f"entropy {metrics.entropy:.2f} < {thresholds.min_entropy}")
    if failures:
        raise QualityCheckError(
            "image quality too low: " + ", ".join(failures), metrics
        )
