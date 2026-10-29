import numpy as np
import pytest

from src.classical.config import RadiomicsConfig
from src.classical.lbp import extract_lbp, feature_names

CONFIG = RadiomicsConfig()


def full_mask(image: np.ndarray) -> np.ndarray:
    return np.ones(image.shape, dtype=bool)


def test_histogram_has_points_plus_two_bins_and_sums_to_one() -> None:
    image = np.random.default_rng(0).random((48, 48))

    features = extract_lbp(image, full_mask(image), CONFIG)

    assert list(features) == feature_names(CONFIG)
    assert len(features) == CONFIG.lbp_points + 2
    assert sum(features.values()) == pytest.approx(1.0)


def test_flat_image_puts_all_mass_in_one_pattern() -> None:
    image = np.full((32, 32), 0.5)

    features = extract_lbp(image, full_mask(image), CONFIG)

    assert max(features.values()) == pytest.approx(1.0)


def test_noise_spreads_over_several_patterns() -> None:
    image = np.random.default_rng(0).random((64, 64))

    features = extract_lbp(image, full_mask(image), CONFIG)

    assert sum(value > 0.01 for value in features.values()) > 3


def test_only_masked_pixels_are_counted() -> None:
    rng = np.random.default_rng(1)
    image = np.zeros((64, 64))
    image[:32] = rng.random((32, 64))
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32] = True

    features = extract_lbp(image, mask, CONFIG)

    assert sum(features.values()) == pytest.approx(1.0)


def test_too_small_mask_gives_zeros() -> None:
    image = np.random.default_rng(0).random((32, 32))
    mask = np.zeros((32, 32), dtype=bool)
    mask[0, :4] = True

    assert set(extract_lbp(image, mask, CONFIG).values()) == {0.0}


def test_result_is_deterministic() -> None:
    image = np.random.default_rng(4).random((32, 32))

    assert extract_lbp(image, full_mask(image), CONFIG) == extract_lbp(
        image, full_mask(image), CONFIG
    )


def test_edge_pixels_do_not_pollute_the_histogram() -> None:
    image = np.full((8, 8), 0.5)
    mask = np.ones((8, 8), dtype=bool)

    features = extract_lbp(image, mask, RadiomicsConfig(min_mask_pixels=4))

    assert max(features.values()) == pytest.approx(1.0)
