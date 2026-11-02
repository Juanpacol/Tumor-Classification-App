"""Checks an uploaded image and removes its embedded metadata before it is stored.

Metadata is removed without re-encoding, so the stored and classified pixels are
exactly the ones that were uploaded.
"""

import io
import struct

from PIL import Image

from domain.errors import FileTooLargeError, UnsupportedMediaTypeError

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

_PILLOW_FORMATS = ["JPEG", "PNG"]

_JPEG_SOI = b"\xff\xd8"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# APP1 holds EXIF and XMP, APP13 holds IPTC, COM is free text.
_JPEG_METADATA_MARKERS = {0xE1, 0xED, 0xFE}
_JPEG_START_OF_SCAN = 0xDA
_JPEG_STANDALONE_MARKERS = {0x01, *range(0xD0, 0xD8)}
_PNG_METADATA_CHUNKS = {b"eXIf", b"tEXt", b"zTXt", b"iTXt", b"tIME"}


def prepare_upload(image: bytes) -> tuple[bytes, str]:
    """Return the image without metadata and its format (``"jpg"`` or ``"png"``).

    The format is read from the content, never from the file name. Raises
    ``FileTooLargeError`` above ``MAX_UPLOAD_BYTES`` and
    ``UnsupportedMediaTypeError`` for anything that is not a well-formed JPEG
    or PNG container, or whose pixels cannot be fully decoded.
    """
    if len(image) > MAX_UPLOAD_BYTES:
        raise FileTooLargeError(MAX_UPLOAD_BYTES)
    if image.startswith(_JPEG_SOI):
        stripped, original_format = _strip_jpeg(image), "jpg"
    elif image.startswith(_PNG_SIGNATURE):
        stripped, original_format = _strip_png(image), "png"
    else:
        raise UnsupportedMediaTypeError()
    _decode(stripped)
    return stripped, original_format


def _decode(image: bytes) -> None:
    try:
        with Image.open(io.BytesIO(image), formats=_PILLOW_FORMATS) as decoded:
            decoded.load()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise UnsupportedMediaTypeError() from None


def _strip_jpeg(image: bytes) -> bytes:
    kept = [_JPEG_SOI]
    position = len(_JPEG_SOI)
    while True:
        if position + 2 > len(image) or image[position] != 0xFF:
            raise UnsupportedMediaTypeError()
        marker = image[position + 1]
        if marker == 0xFF:
            position += 1
            continue
        if marker in _JPEG_STANDALONE_MARKERS:
            kept.append(image[position : position + 2])
            position += 2
            continue
        if position + 4 > len(image):
            raise UnsupportedMediaTypeError()
        (length,) = struct.unpack(">H", image[position + 2 : position + 4])
        end = position + 2 + length
        if length < 2 or end > len(image):
            raise UnsupportedMediaTypeError()
        if marker == _JPEG_START_OF_SCAN:
            kept.append(image[position:])
            return b"".join(kept)
        if marker not in _JPEG_METADATA_MARKERS:
            kept.append(image[position:end])
        position = end


def _strip_png(image: bytes) -> bytes:
    kept = [_PNG_SIGNATURE]
    position = len(_PNG_SIGNATURE)
    while position < len(image):
        if position + 8 > len(image):
            raise UnsupportedMediaTypeError()
        (length,) = struct.unpack(">I", image[position : position + 4])
        kind = image[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(image):
            raise UnsupportedMediaTypeError()
        if kind not in _PNG_METADATA_CHUNKS:
            kept.append(image[position:end])
        position = end
        if kind == b"IEND":
            return b"".join(kept)
    raise UnsupportedMediaTypeError()
