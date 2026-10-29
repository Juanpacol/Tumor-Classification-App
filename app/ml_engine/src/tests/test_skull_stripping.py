import numpy as np

from src.preprocessing.skull_stripping import (
    MAX_MASK_FRACTION,
    MIN_MASK_FRACTION,
    strip_skull,
)
from src.tests.synthetic import head_phantom


def test_scalp_is_removed_and_brain_is_kept() -> None:
    size = 256
    phantom = head_phantom(size)

    result = strip_skull(phantom)

    assert result.applied is True
    assert MIN_MASK_FRACTION <= result.mask.mean() <= MAX_MASK_FRACTION
    assert result.image[size // 2, size // 2] == phantom[size // 2, size // 2]
    scalp_row, scalp_col = size // 2, int(size / 2 + 0.4 * size)
    assert phantom[scalp_row, scalp_col] > 0
    assert result.image[scalp_row, scalp_col] == 0.0
    assert result.image.dtype == np.float32


def test_flat_image_falls_back_to_the_input() -> None:
    flat = np.full((128, 128), 0.5, dtype=np.float32)

    result = strip_skull(flat)

    assert result.applied is False
    assert result.mask.all()
    assert np.array_equal(result.image, flat)


def test_implausibly_small_mask_falls_back() -> None:
    yy, xx = np.mgrid[:256, :256]
    small = (np.hypot(yy - 128, xx - 128) < 25).astype(np.float32)

    result = strip_skull(small)

    assert result.applied is False
    assert np.array_equal(result.image, small)


def test_noise_never_raises() -> None:
    noise = np.random.default_rng(0).random((128, 128)).astype(np.float32)

    result = strip_skull(noise)

    assert result.mask.shape == noise.shape
