import numpy as np
from PIL import Image

from src.preprocessing.deduplication.hashing import (
    hamming_distances,
    perceptual_hash,
    to_gray_uint8,
)
from src.tests.synthetic import synthetic_slice


def test_hamming_distances() -> None:
    hashes = np.array([0b0000, 0b0001, 0b1111, 2**64 - 1], dtype=np.uint64)

    assert hamming_distances(0, hashes).tolist() == [0, 1, 4, 64]


def test_perceptual_hash_is_robust_to_resize_but_separates_different_images() -> None:
    original = synthetic_slice(1)
    others = np.array(
        [
            perceptual_hash(original.resize((128, 128), Image.Resampling.BILINEAR)),
            perceptual_hash(synthetic_slice(2)),
        ],
        dtype=np.uint64,
    )

    near, far = hamming_distances(perceptual_hash(original), others).tolist()

    assert near <= 4
    assert far > 12


def test_sixteen_bit_images_are_rescaled_not_clipped() -> None:
    pixels = np.array([[0, 1000], [30000, 60000]], dtype=np.uint16)

    gray = to_gray_uint8(Image.fromarray(pixels))

    assert gray.tolist() == [[0, 4], [128, 255]]
