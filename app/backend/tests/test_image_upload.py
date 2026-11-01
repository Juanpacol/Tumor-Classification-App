import io

import pytest
from PIL import Image

from application.image_upload import MAX_UPLOAD_BYTES, prepare_upload
from domain.errors import FileTooLargeError, UnsupportedMediaTypeError
from tests.factories import image_bytes


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
