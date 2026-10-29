import time

import numpy as np
import pytest

from src.classical.config import RadiomicsConfig
from src.classical.extractor import extract_radiomics, feature_names
from src.tests.synthetic import head_phantom, synthetic_slice


def full_mask(image: np.ndarray) -> np.ndarray:
    return np.ones(image.shape, dtype=bool)


def test_vector_has_45_unique_ordered_names() -> None:
    names = feature_names()

    assert len(names) == 45
    assert len(set(names)) == 45
    assert names[0].startswith("glcm_") and names[-1].startswith("intensity_")


def test_names_and_length_are_stable_across_inputs() -> None:
    rng = np.random.default_rng(0)
    inputs = [
        (rng.random((64, 64)), np.ones((64, 64), dtype=bool)),
        (head_phantom(128), head_phantom(128) > 0),
        (np.full((64, 64), 0.3), np.ones((64, 64), dtype=bool)),
    ]

    for image, mask in inputs:
        result = extract_radiomics(image, mask)

        assert result.names == feature_names()
        assert result.values.shape == (45,)


@pytest.mark.parametrize(
    "case",
    ["constant", "noise", "tiny_mask", "empty_mask", "single_pixel_line", "black"],
)
def test_output_is_always_finite(case: str) -> None:
    rng = np.random.default_rng(0)
    image = rng.random((64, 64))
    mask = np.ones((64, 64), dtype=bool)
    if case == "constant":
        image = np.full((64, 64), 0.5)
    elif case == "tiny_mask":
        mask = np.zeros((64, 64), dtype=bool)
        mask[:2, :3] = True
    elif case == "empty_mask":
        mask = np.zeros((64, 64), dtype=bool)
    elif case == "single_pixel_line":
        mask = np.zeros((64, 64), dtype=bool)
        mask[20, :] = True
    elif case == "black":
        image = np.zeros((64, 64))

    assert np.isfinite(extract_radiomics(image, mask).values).all()


def test_result_is_deterministic() -> None:
    image = np.asarray(synthetic_slice(7), dtype=np.float64) / 255.0

    first = extract_radiomics(image, image > 0.1)
    second = extract_radiomics(image, image > 0.1)

    assert np.array_equal(first.values, second.values)


def test_config_changes_the_vector_length() -> None:
    config = RadiomicsConfig(lbp_points=16, wavelet_level=3)
    image = np.random.default_rng(0).random((64, 64))

    result = extract_radiomics(image, full_mask(image), config)

    assert result.names == feature_names(config)
    assert len(result.names) == 5 + 18 + 10 * 3 + 9


def test_invalid_inputs_are_rejected() -> None:
    image = np.random.default_rng(0).random((32, 32))

    with pytest.raises(ValueError, match="2D"):
        extract_radiomics(image[None], np.ones((1, 32, 32), dtype=bool))
    with pytest.raises(ValueError, match="boolean"):
        extract_radiomics(image, np.ones((32, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="same shape"):
        extract_radiomics(image, np.ones((16, 16), dtype=bool))


def test_extraction_takes_under_a_second_on_a_512_image() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((512, 512))
    mask = np.ones((512, 512), dtype=bool)
    extract_radiomics(image, mask)

    start = time.perf_counter()
    extract_radiomics(image, mask)

    assert time.perf_counter() - start < 1.0
