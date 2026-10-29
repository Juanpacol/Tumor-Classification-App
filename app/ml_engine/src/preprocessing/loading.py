"""Loading of 2D brain MRI slices from JPG/PNG files."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.errors import UnsupportedImageError
from src.preprocessing.deduplication.hashing import to_gray_uint8

MIN_SIDE = 64
MAX_PIXELS = 25_000_000
MAX_FILE_BYTES = 50 * 1024 * 1024
_ALLOWED_FORMATS = ["PNG", "JPEG"]
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_DICOM_OFFSET = 128
_DICOM_MAGIC = b"DICM"


def load_mri_image(path: str | Path) -> np.ndarray:
    """Load a JPG/PNG file as a 2D ``float32`` grayscale array scaled to [0, 1].

    The file is read once, up to ``MAX_FILE_BYTES``. See ``decode_mri_image``
    for the checks and for the errors it raises; an unreadable file also
    raises ``UnsupportedImageError``.
    """
    path = Path(path)
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise UnsupportedImageError(f"cannot read image: {path}") from exc
    return decode_mri_image(data, str(path))


def decode_mri_image(data: bytes, source: str = "image") -> np.ndarray:
    """Decode JPG/PNG bytes as a 2D ``float32`` grayscale array scaled to [0, 1].

    The format is detected from the content, not from a file name, and the EXIF
    orientation is applied. Raises ``UnsupportedImageError`` for DICOM and any
    other format, for corrupt data, for more than ``MAX_FILE_BYTES``, for more
    than ``MAX_PIXELS`` and for images under ``MIN_SIDE`` pixels on either side.
    ``source`` names the input in the error messages.
    """
    if len(data) > MAX_FILE_BYTES:
        raise UnsupportedImageError(f"file exceeds {MAX_FILE_BYTES} bytes")
    if data[_DICOM_OFFSET : _DICOM_OFFSET + len(_DICOM_MAGIC)] == _DICOM_MAGIC:
        raise UnsupportedImageError("DICOM is not supported, use JPG or PNG")
    if not data.startswith((_PNG_MAGIC, _JPEG_MAGIC)):
        raise UnsupportedImageError("unsupported format, use JPG or PNG")

    try:
        with Image.open(BytesIO(data), formats=_ALLOWED_FORMATS) as image:
            width, height = image.size
            if min(width, height) < MIN_SIDE:
                raise UnsupportedImageError(
                    f"image is {width}x{height}, minimum is {MIN_SIDE}x{MIN_SIDE}"
                )
            if width * height > MAX_PIXELS:
                raise UnsupportedImageError(
                    f"image is {width}x{height}, maximum is {MAX_PIXELS} pixels"
                )
            image.load()
            gray = to_gray_uint8(ImageOps.exif_transpose(image))
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise UnsupportedImageError(f"corrupt or unreadable image: {source}") from exc

    return gray.astype(np.float32) / 255.0
