import numpy as np
import pytest

from src.classical.config import RadiomicsConfig
from src.classical.wavelet import extract_wavelet, feature_names

CONFIG = RadiomicsConfig()


def full_mask(image: np.ndarray) -> np.ndarray:
    return np.ones(image.shape, dtype=bool)


def test_level_two_has_seven_bands_with_three_statistics() -> None:
    image = np.random.default_rng(0).random((64, 64))

    features = extract_wavelet(image, full_mask(image), CONFIG)

    assert list(features) == feature_names(CONFIG)
    assert len(features) == 21
    assert feature_names(CONFIG)[0] == "wavelet_cA2_energy"
    assert feature_names(CONFIG)[-1] == "wavelet_cD1_std"


def test_flat_image_has_no_detail_energy() -> None:
    image = np.full((64, 64), 0.5)

    features = extract_wavelet(image, full_mask(image), CONFIG)

    assert features["wavelet_cA2_energy"] > 0
    for band in ("cH2", "cV2", "cD2", "cH1", "cV1", "cD1"):
        assert features[f"wavelet_{band}_energy"] == pytest.approx(0.0, abs=1e-12)


def test_noise_has_more_detail_energy_than_a_smooth_ramp() -> None:
    noise = np.random.default_rng(0).random((64, 64))
    ramp = np.tile(np.linspace(0, 1, 64), (64, 1))

    noisy = extract_wavelet(noise, full_mask(noise), CONFIG)["wavelet_cD1_energy"]
    smooth = extract_wavelet(ramp, full_mask(ramp), CONFIG)["wavelet_cD1_energy"]

    assert noisy > smooth


def test_odd_sized_images_and_masks_are_handled() -> None:
    image = np.random.default_rng(1).random((63, 77))
    mask = np.zeros(image.shape, dtype=bool)
    mask[5:40, 10:60] = True

    features = extract_wavelet(image, mask, CONFIG)

    assert all(np.isfinite(list(features.values())))


def test_too_small_mask_gives_zeros() -> None:
    image = np.random.default_rng(0).random((32, 32))
    mask = np.zeros((32, 32), dtype=bool)
    mask[0, :4] = True

    assert set(extract_wavelet(image, mask, CONFIG).values()) == {0.0}


def test_result_is_deterministic() -> None:
    image = np.random.default_rng(5).random((48, 48))

    assert extract_wavelet(image, full_mask(image), CONFIG) == extract_wavelet(
        image, full_mask(image), CONFIG
    )


def test_region_too_thin_for_the_level_gives_zeros_without_warnings(
    recwarn: pytest.WarningsRecorder,
) -> None:
    image = np.random.default_rng(0).random((64, 64))
    mask = np.zeros((64, 64), dtype=bool)
    mask[20, :] = True

    assert set(extract_wavelet(image, mask, CONFIG).values()) == {0.0}
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
