"""Image hashing primitives: grayscale conversion, pHash, Hamming distance, thumbnails."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fft import dct

_HASH_SIDE = 8
_THUMBNAIL_SIDE = 32
_POPCOUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def to_gray_uint8(image: Image.Image) -> np.ndarray:
    """Convert to 8-bit grayscale, min-max rescaling high bit-depth modes instead of clipping."""
    if not (image.mode.startswith("I") or image.mode == "F"):
        return np.asarray(image.convert("L"))
    pixels = np.asarray(image, dtype=np.float64)
    low, high = pixels.min(), pixels.max()
    if high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)
    return np.round((pixels - low) * (255.0 / (high - low))).astype(np.uint8)


def perceptual_hash(image: Image.Image) -> int:
    """64-bit DCT pHash: the 8x8 lowest frequencies of a 32x32 downscale, thresholded at the median."""
    side = _HASH_SIDE * 4
    pixels = np.asarray(
        image.convert("L").resize((side, side), Image.Resampling.LANCZOS), dtype=np.float64
    )
    low_frequencies = dct(dct(pixels, axis=0), axis=1)[:_HASH_SIDE, :_HASH_SIDE]
    bits = (low_frequencies > np.median(low_frequencies)).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def hamming_distances(reference: int, others: np.ndarray) -> np.ndarray:
    """Hamming distance between a 64-bit hash and an array of ``uint64`` hashes."""
    xor = np.bitwise_xor(others, np.uint64(reference))
    return _POPCOUNT[xor.view(np.uint8)].reshape(-1, 8).sum(axis=1, dtype=np.int64)


def normalized_thumbnail(image: Image.Image) -> np.ndarray:
    """32x32 grayscale thumbnail standardized to zero mean and unit variance."""
    resized = image.convert("L").resize(
        (_THUMBNAIL_SIDE, _THUMBNAIL_SIDE), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(resized, dtype=np.float32)
    return (pixels - pixels.mean()) / (pixels.std() + 1e-8)
