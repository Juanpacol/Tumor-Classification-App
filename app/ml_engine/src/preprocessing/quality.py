"""Image quality measurements and the quality gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.filters import threshold_otsu
from skimage.restoration import estimate_sigma

from src.contracts import QualityMetrics
from src.errors import QualityCheckError

_HISTOGRAM_BINS = 256
_EPSILON = 1e-6


@dataclass(frozen=True)
class QualityThresholds:
    """Minimum acceptable quality.

    The defaults are provisional. Calibration on the purged training split
    was run and recorded but not applied, because the current metrics reject
    usable scans (see docs/ml-engine/decisions.md, D15).
    """

    min_snr: float = 4.0
    min_entropy: float = 2.0


def shannon_entropy(gray: np.ndarray) -> float:
    """Shannon entropy in bits of the intensity histogram of a [0, 1] image."""
    counts, _ = np.histogram(gray, bins=_HISTOGRAM_BINS, range=(0.0, 1.0))
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def estimate_snr(gray: np.ndarray) -> float:
    """Mean foreground intensity divided by the estimated noise sigma.

    Foreground is everything above the Otsu threshold and sigma comes from a
    wavelet estimator, so no noise-only background region is needed. A flat
    image has no signal and returns 0.
    """
    if gray.max() <= gray.min():
        return 0.0
    foreground = gray > threshold_otsu(gray)
    if not foreground.any():
        return 0.0
    sigma = float(estimate_sigma(gray))
    return float(gray[foreground].mean() / max(sigma, _EPSILON))


def measure_quality(gray: np.ndarray, skull_stripped: bool) -> QualityMetrics:
    """Measure SNR and entropy of a 2D [0, 1] image."""
    return QualityMetrics(
        snr=estimate_snr(gray),
        entropy=shannon_entropy(gray),
        skull_stripped=skull_stripped,
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
