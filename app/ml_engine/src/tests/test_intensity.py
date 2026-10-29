import numpy as np
import pytest
from scipy import stats

from src.classical.config import RadiomicsConfig
from src.classical.intensity import extract_intensity, feature_names

CONFIG = RadiomicsConfig()


def full_mask(image: np.ndarray) -> np.ndarray:
    return np.ones(image.shape, dtype=bool)


def test_values_match_numpy_and_scipy() -> None:
    image = np.random.default_rng(0).random((32, 32))

    features = extract_intensity(image, full_mask(image), CONFIG)

    assert features["intensity_mean"] == pytest.approx(image.mean())
    assert features["intensity_std"] == pytest.approx(image.std())
    assert features["intensity_skewness"] == pytest.approx(stats.skew(image.ravel()))
    assert features["intensity_kurtosis"] == pytest.approx(
        stats.kurtosis(image.ravel())
    )
    assert features["intensity_p50"] == pytest.approx(np.median(image))


def test_keys_follow_feature_names() -> None:
    image = np.random.default_rng(0).random((32, 32))

    assert list(extract_intensity(image, full_mask(image), CONFIG)) == feature_names(
        CONFIG
    )


def test_only_masked_pixels_are_used() -> None:
    image = np.zeros((32, 32))
    image[:16] = 0.8
    mask = np.zeros((32, 32), dtype=bool)
    mask[:16] = True
    mask[0, 0] = False

    features = extract_intensity(image, mask, CONFIG)

    assert features["intensity_mean"] == pytest.approx(0.8)
    assert features["intensity_p10"] == pytest.approx(0.8)


def test_flat_region_has_zero_spread_and_finite_values() -> None:
    image = np.full((32, 32), 0.4)

    features = extract_intensity(image, full_mask(image), CONFIG)

    assert features["intensity_mean"] == pytest.approx(0.4)
    assert features["intensity_std"] == 0.0
    assert features["intensity_skewness"] == 0.0
    assert features["intensity_kurtosis"] == 0.0
    assert all(np.isfinite(list(features.values())))


def test_too_small_mask_gives_zeros() -> None:
    image = np.random.default_rng(0).random((32, 32))
    mask = np.zeros((32, 32), dtype=bool)
    mask[0, :4] = True

    assert set(extract_intensity(image, mask, CONFIG).values()) == {0.0}


def test_percentiles_are_ordered() -> None:
    image = np.random.default_rng(1).random((32, 32))

    features = extract_intensity(image, full_mask(image), CONFIG)

    percentiles = [features[f"intensity_p{p}"] for p in CONFIG.percentiles]
    assert percentiles == sorted(percentiles)
