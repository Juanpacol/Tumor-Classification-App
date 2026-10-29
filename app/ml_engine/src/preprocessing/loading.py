"""Loading of 2D brain MRI slices from JPG/PNG files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.errors import UnsupportedImageError
from src.preprocessing.deduplication.hashing import to_gray_uint8

MIN_SIDE = 64
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_DICOM_OFFSET = 128
_DICOM_MAGIC = b"DICM"
_HEADER_SIZE = _DICOM_OFFSET + len(_DICOM_MAGIC)


def load_mri_image(path: str | Path) -> np.ndarray:
    """Load a JPG/PNG as a 2D ``float32`` grayscale array scaled to [0, 1].

    The format is detected from the file content, not the extension.
    Raises ``UnsupportedImageError`` for DICOM and any other format, for
    unreadable or corrupt files and for images under ``MIN_SIDE`` pixels on
    either side.
    """
    path = Path(path)
    try:
        with path.open("rb") as handle:
            header = handle.read(_HEADER_SIZE)
    except OSError as exc:
        raise UnsupportedImageError(f"cannot read image: {path}") from exc

    if header[_DICOM_OFFSET:] == _DICOM_MAGIC:
        raise UnsupportedImageError("DICOM is not supported, use JPG or PNG")
    if not header.startswith((_PNG_MAGIC, _JPEG_MAGIC)):
        raise UnsupportedImageError("unsupported format, use JPG or PNG")

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if min(width, height) < MIN_SIDE:
                raise UnsupportedImageError(
                    f"image is {width}x{height}, minimum is {MIN_SIDE}x{MIN_SIDE}"
                )
            gray = to_gray_uint8(image)
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise UnsupportedImageError(f"corrupt or unreadable image: {path}") from exc

    return gray.astype(np.float32) / 255.0
