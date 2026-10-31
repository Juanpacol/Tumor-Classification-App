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
from src.preprocessing.skull_stripping import foreground_mask
from src.tests.synthetic import synthetic_slice


def slice_array(seed: int = 1) -> np.ndarray:
    return np.asarray(synthetic_slice(seed), dtype=np.float32) / 255.0


def noisy_background_slice() -> np.ndarray:
    """Synthetic slice whose black background carries mild Gaussian noise."""
    noise = np.random.default_rng(3).normal(0, 0.02, (256, 256))
    return np.clip(slice_array() + np.abs(noise), 0, 1).astype(np.float32)


def snr(gray: np.ndarray) -> float:
    return measure_quality(gray, False).snr


def test_entropy_of_flat_image_is_zero() -> None:
    flat = np.full((64, 64), 0.5, dtype=np.float32)

    assert shannon_entropy(flat, np.ones(flat.shape, dtype=bool)) == 0.0
    assert measure_quality(flat, False).entropy == 0.0


def test_entropy_of_empty_mask_is_zero() -> None:
    gray = slice_array()

    assert shannon_entropy(gray, np.zeros(gray.shape, dtype=bool)) == 0.0


def test_entropy_of_uniform_noise_is_close_to_eight_bits() -> None:
    noise = np.random.default_rng(0).random((256, 256)).astype(np.float32)

    assert measure_quality(noise, False).entropy == pytest.approx(8.0, abs=0.05)


def test_black_padding_does_not_change_entropy() -> None:
    gray = slice_array()
    padded = np.pad(gray, ((64, 64), (64, 64)))

    assert padded.size >= 2 * gray.size
    assert measure_quality(padded, False).entropy == pytest.approx(
        measure_quality(gray, False).entropy, abs=0.05
    )


def test_snr_of_flat_image_is_zero() -> None:
    assert snr(np.zeros((64, 64), dtype=np.float32)) == 0.0
    assert snr(np.full((64, 64), 0.5, dtype=np.float32)) == 0.0


def test_snr_drops_when_noise_is_added() -> None:
    clean = noisy_background_slice()
    noisy = np.clip(clean + np.random.default_rng(0).normal(0, 0.1, clean.shape), 0, 1)

    assert snr(noisy.astype(np.float32)) < snr(clean)


def test_zero_background_gives_median_over_one_8bit_step() -> None:
    gray = slice_array()
    mask = foreground_mask(gray)

    assert estimate_snr(gray, mask) == pytest.approx(np.median(gray[mask]) * 255)


def test_texture_inside_the_brain_does_not_change_snr() -> None:
    gray = noisy_background_slice()
    mask = foreground_mask(gray)
    texture = np.random.default_rng(5).choice([-0.03, 0.03], size=gray.shape)
    textured = np.where(mask, np.clip(gray + texture, 0, 1), gray).astype(np.float32)

    assert estimate_snr(textured, foreground_mask(textured)) == pytest.approx(
        estimate_snr(gray, mask), rel=0.05
    )


def test_tissue_outside_the_brain_does_not_change_snr() -> None:
    gray = noisy_background_slice()
    with_neck = gray.copy()
    with_neck[236:252, 100:156] = 0.6

    assert snr(with_neck) == pytest.approx(snr(gray), rel=0.05)


def test_frame_filling_image_has_a_finite_snr() -> None:
    gray = np.asarray(synthetic_slice(2), dtype=np.float32)[60:196, 60:196] / 255.0

    value = snr(gray)

    assert np.isfinite(value) and value >= 0


def test_uniform_noise_has_snr_below_two() -> None:
    noise = np.random.default_rng(1).random((256, 256)).astype(np.float32)

    assert snr(noise) < 2.0


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
