from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.errors import UnsupportedImageError
from src.preprocessing import loading
from src.preprocessing.loading import MAX_PIXELS, MIN_SIDE, load_mri_image
from src.tests.synthetic import save, synthetic_slice


def assert_valid(array: np.ndarray, shape: tuple[int, int]) -> None:
    assert array.dtype == np.float32
    assert array.shape == shape
    assert 0.0 <= array.min() <= array.max() <= 1.0


def test_grayscale_png_is_loaded(tmp_path: Path) -> None:
    path = save(synthetic_slice(1, size=128), tmp_path / "a.png")

    assert_valid(load_mri_image(path), (128, 128))


def test_rgb_png_is_converted_to_grayscale(tmp_path: Path) -> None:
    path = save(synthetic_slice(1, size=96).convert("RGB"), tmp_path / "a.png")

    assert_valid(load_mri_image(path), (96, 96))


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_jpeg_is_loaded(tmp_path: Path, mode: str) -> None:
    path = save(synthetic_slice(2, size=100).convert(mode), tmp_path / "a.jpg")

    assert_valid(load_mri_image(path), (100, 100))


def test_16_bit_png_is_rescaled_not_clipped(tmp_path: Path) -> None:
    pixels = np.linspace(0, 60000, 128 * 128, dtype=np.uint16).reshape(128, 128)
    path = save(Image.fromarray(pixels), tmp_path / "deep.png")

    array = load_mri_image(path)

    assert_valid(array, (128, 128))
    assert array.max() == pytest.approx(1.0)
    assert array.min() == pytest.approx(0.0)


def test_extension_does_not_decide_the_format(tmp_path: Path) -> None:
    path = tmp_path / "actually_png.jpg"
    synthetic_slice(3, size=80).save(path, format="PNG")

    assert_valid(load_mri_image(path), (80, 80))


def test_dicom_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "scan.png"
    path.write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 64)

    with pytest.raises(UnsupportedImageError, match="DICOM"):
        load_mri_image(path)


@pytest.mark.parametrize("fmt", ["BMP", "GIF", "TIFF"])
def test_other_formats_are_rejected_even_with_png_extension(
    tmp_path: Path, fmt: str
) -> None:
    path = tmp_path / "scan.png"
    synthetic_slice(4, size=128).save(path, format=fmt)

    with pytest.raises(UnsupportedImageError, match="unsupported format"):
        load_mri_image(path)


def test_truncated_png_is_rejected(tmp_path: Path) -> None:
    good = save(synthetic_slice(5, size=128), tmp_path / "good.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(good.read_bytes()[:200])

    with pytest.raises(UnsupportedImageError, match="corrupt"):
        load_mri_image(bad)


def test_png_header_with_garbage_body_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not an image at all")

    with pytest.raises(UnsupportedImageError, match="corrupt"):
        load_mri_image(path)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.png"
    path.write_bytes(b"")

    with pytest.raises(UnsupportedImageError, match="unsupported format"):
        load_mri_image(path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedImageError, match="cannot read"):
        load_mri_image(tmp_path / "missing.png")


@pytest.mark.parametrize(
    ("width", "height"), [(MIN_SIDE - 1, MIN_SIDE), (MIN_SIDE, MIN_SIDE - 1), (32, 32)]
)
def test_image_smaller_than_minimum_is_rejected(
    tmp_path: Path, width: int, height: int
) -> None:
    pixels = np.full((height, width), 128, dtype=np.uint8)
    path = save(Image.fromarray(pixels), tmp_path / "small.png")

    with pytest.raises(UnsupportedImageError, match="minimum"):
        load_mri_image(path)


def test_minimum_size_is_accepted(tmp_path: Path) -> None:
    pixels = np.full((MIN_SIDE, MIN_SIDE), 128, dtype=np.uint8)
    path = save(Image.fromarray(pixels), tmp_path / "edge.png")

    assert_valid(load_mri_image(path), (MIN_SIDE, MIN_SIDE))


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6
    pixels = np.full((96, 128), 128, dtype=np.uint8)
    path = save(Image.fromarray(pixels), tmp_path / "rotated.jpg", exif=exif)

    assert_valid(load_mri_image(path), (128, 96))


def test_image_over_pixel_limit_is_rejected_before_decoding(tmp_path: Path) -> None:
    side = int(MAX_PIXELS**0.5) + 1
    path = save(Image.new("L", (side, side)), tmp_path / "huge.png")

    with pytest.raises(UnsupportedImageError, match="maximum"):
        load_mri_image(path)


def test_file_over_byte_limit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = save(synthetic_slice(6, size=128), tmp_path / "a.png")
    monkeypatch.setattr(loading, "MAX_FILE_BYTES", path.stat().st_size - 1)

    with pytest.raises(UnsupportedImageError, match="exceeds"):
        load_mri_image(path)
