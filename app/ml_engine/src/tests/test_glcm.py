import numpy as np
import pytest

from src.classical.config import RadiomicsConfig
from src.classical.glcm import extract_glcm, feature_names

CONFIG = RadiomicsConfig()


def checkerboard(size: int = 32) -> np.ndarray:
    return ((np.indices((size, size)).sum(axis=0) % 2) * 0.9 + 0.05).astype(np.float64)


def ramp(size: int = 32) -> np.ndarray:
    return np.tile(np.linspace(0.05, 0.95, size), (size, 1))


def full_mask(image: np.ndarray) -> np.ndarray:
    return np.ones(image.shape, dtype=bool)


def test_keys_follow_feature_names() -> None:
    image = np.random.default_rng(0).random((32, 32))

    assert list(extract_glcm(image, full_mask(image), CONFIG)) == feature_names(CONFIG)


def test_coarse_texture_has_higher_contrast_than_a_smooth_ramp() -> None:
    contrast = {
        name: extract_glcm(image, full_mask(image), CONFIG)["glcm_contrast"]
        for name, image in (("checker", checkerboard()), ("ramp", ramp()))
    }

    assert contrast["checker"] > contrast["ramp"]


def test_smooth_ramp_is_more_homogeneous_than_noise() -> None:
    noise = np.random.default_rng(0).random((32, 32))

    smooth = extract_glcm(ramp(), full_mask(ramp()), CONFIG)["glcm_homogeneity"]
    rough = extract_glcm(noise, full_mask(noise), CONFIG)["glcm_homogeneity"]

    assert smooth > rough


def test_background_does_not_change_the_features() -> None:
    texture = np.random.default_rng(2).random((40, 40))
    padded = np.zeros((120, 120))
    padded[30:70, 40:80] = texture
    padded_mask = np.zeros((120, 120), dtype=bool)
    padded_mask[30:70, 40:80] = True

    alone = extract_glcm(texture, full_mask(texture), CONFIG)
    with_background = extract_glcm(padded, padded_mask, CONFIG)

    assert with_background == pytest.approx(alone)


def test_flat_region_gives_zeros() -> None:
    image = np.full((32, 32), 0.5)

    assert set(extract_glcm(image, full_mask(image), CONFIG).values()) == {0.0}


def test_too_small_mask_gives_zeros() -> None:
    image = np.random.default_rng(0).random((32, 32))
    mask = np.zeros((32, 32), dtype=bool)
    mask[0, :4] = True

    assert set(extract_glcm(image, mask, CONFIG).values()) == {0.0}


def test_thin_one_pixel_line_is_finite() -> None:
    image = np.random.default_rng(0).random((32, 32))
    mask = np.zeros((32, 32), dtype=bool)
    mask[10, :] = True

    assert all(np.isfinite(list(extract_glcm(image, mask, CONFIG).values())))


def test_result_is_deterministic() -> None:
    image = np.random.default_rng(3).random((32, 32))

    first = extract_glcm(image, full_mask(image), CONFIG)

    assert first == extract_glcm(image, full_mask(image), CONFIG)
