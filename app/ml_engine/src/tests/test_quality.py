import numpy as np
import pytest

from src.contracts import QualityMetrics
from src.errors import QualityCheckError
from src.preprocessing.quality import (
    QualityThresholds,
    check_quality,
    estimate_snr,
    measure_quality,
    shannon_entropy,
)
from src.tests.synthetic import synthetic_slice


def slice_array(seed: int = 1) -> np.ndarray:
    return np.asarray(synthetic_slice(seed), dtype=np.float32) / 255.0


def test_entropy_of_flat_image_is_zero() -> None:
    assert shannon_entropy(np.full((64, 64), 0.5, dtype=np.float32)) == 0.0


def test_entropy_of_uniform_noise_is_close_to_eight_bits() -> None:
    noise = np.random.default_rng(0).random((256, 256)).astype(np.float32)

    assert shannon_entropy(noise) == pytest.approx(8.0, abs=0.05)


def test_snr_of_flat_image_is_zero() -> None:
    assert estimate_snr(np.zeros((64, 64), dtype=np.float32)) == 0.0


def test_snr_drops_when_noise_is_added() -> None:
    clean = slice_array()
    noisy = np.clip(clean + np.random.default_rng(0).normal(0, 0.2, clean.shape), 0, 1)

    assert estimate_snr(noisy.astype(np.float32)) < estimate_snr(clean)


def test_measure_quality_reports_the_skull_stripping_flag() -> None:
    metrics = measure_quality(slice_array(), skull_stripped=True)

    assert metrics.skull_stripped is True
    assert metrics.snr > 0 and metrics.entropy > 0


def test_synthetic_slice_passes_the_gate() -> None:
    check_quality(measure_quality(slice_array(), False), QualityThresholds())


def test_blank_image_is_rejected_with_measured_values() -> None:
    blank = np.zeros((128, 128), dtype=np.float32)
    metrics = measure_quality(blank, False)

    with pytest.raises(QualityCheckError) as error:
        check_quality(metrics, QualityThresholds())

    assert error.value.metrics == metrics
    assert "snr" in error.value.reason and "entropy" in error.value.reason


def test_pure_noise_is_rejected_by_snr() -> None:
    noise = np.random.default_rng(1).random((128, 128)).astype(np.float32)

    with pytest.raises(QualityCheckError, match="snr"):
        check_quality(measure_quality(noise, False), QualityThresholds())


def test_thresholds_are_configurable() -> None:
    metrics = QualityMetrics(snr=3.0, entropy=5.0, skull_stripped=False)

    with pytest.raises(QualityCheckError):
        check_quality(metrics, QualityThresholds(min_snr=4.0))
    check_quality(metrics, QualityThresholds(min_snr=2.0))
