import io
import struct
import zlib

import pytest
from PIL import Image

from application.image_upload import MAX_UPLOAD_BYTES, prepare_upload
from domain.errors import FileTooLargeError, UnsupportedMediaTypeError
from tests.factories import image_bytes, patterned_image


@pytest.mark.parametrize("fmt, expected", [("PNG", "png"), ("JPEG", "jpg")])
def test_the_format_is_read_from_the_content(fmt, expected):
    _, original_format = prepare_upload(image_bytes(fmt))

    assert original_format == expected


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_metadata_is_removed_and_the_pixels_are_kept(fmt):
    original = image_bytes(fmt, with_metadata=True)
    assert b"Jane Doe" in original

    stripped, _ = prepare_upload(original)

    assert b"Jane Doe" not in stripped
    with (
        Image.open(io.BytesIO(stripped)) as clean,
        Image.open(io.BytesIO(original)) as source,
    ):
        assert not clean.getexif()
        assert clean.tobytes() == source.tobytes()


def test_an_image_without_metadata_is_unchanged():
    original = image_bytes("PNG")

    assert prepare_upload(original)[0] == original


def test_a_file_over_the_limit_is_too_large():
    with pytest.raises(FileTooLargeError) as error:
        prepare_upload(b"\x89PNG" + b"0" * MAX_UPLOAD_BYTES)

    assert error.value.status_code == 413


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"GIF89a....",
        b"DICM" + b"\x00" * 128,
        image_bytes("PNG")[:40],
        image_bytes("JPEG")[:12],
    ],
    ids=["empty", "gif", "dicom", "truncated-png", "truncated-jpeg"],
)
def test_anything_but_a_well_formed_jpeg_or_png_is_unsupported(content):
    with pytest.raises(UnsupportedMediaTypeError) as error:
        prepare_upload(content)

    assert error.value.status_code == 415


def _with_corrupt_idat(png: bytes) -> bytes:
    start = png.index(b"IDAT") - 4
    (length,) = struct.unpack(">I", png[start : start + 4])
    data = bytes(b ^ 0xFF for b in png[start + 8 : start + 8 + length])
    crc = struct.pack(">I", zlib.crc32(b"IDAT" + data))
    return png[: start + 8] + data + crc + png[start + 12 + length :]


@pytest.mark.parametrize(
    "content",
    [patterned_image("JPEG")[:-200], _with_corrupt_idat(patterned_image("PNG"))],
    ids=["truncated-jpeg-scan", "corrupt-png-idat"],
)
def test_a_well_formed_container_with_undecodable_pixels_is_unsupported(content):
    with pytest.raises(UnsupportedMediaTypeError) as error:
        prepare_upload(content)

    assert error.value.status_code == 415


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_a_decodable_image_passes(fmt):
    original = patterned_image(fmt)

    assert prepare_upload(original)[0] == original
