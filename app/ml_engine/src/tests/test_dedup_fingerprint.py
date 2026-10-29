from pathlib import Path

import pytest
from PIL import Image

from src.preprocessing.deduplication.discovery import discover_images
from src.preprocessing.deduplication.fingerprint import (
    compute_fingerprint,
    is_augmented_name,
    thumbnail_correlation,
)
from src.tests.synthetic import save, synthetic_slice


def test_is_augmented_name() -> None:
    assert is_augmented_name(Path("Tr-aug-me_12.jpg"))
    assert is_augmented_name(Path("img_aug_3.png"))
    assert not is_augmented_name(Path("Tr-gl_12.jpg"))
    assert not is_augmented_name(Path("augusto.png"))


def test_correlation_is_invariant_to_contrast_and_mirroring(tmp_path: Path) -> None:
    original = synthetic_slice(3)
    variants = {
        "o.png": original,
        "r.png": original.point(lambda v: int(v * 0.6) + 30),
        "m.png": original.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "x.png": synthetic_slice(4),
    }
    o, r, m, x = (
        compute_fingerprint(save(img, tmp_path / name), "Train", "glioma")
        for name, img in variants.items()
    )

    assert thumbnail_correlation(o, r) > 0.99
    assert thumbnail_correlation(o, m) > 0.99
    assert thumbnail_correlation(o, x) < 0.95


def test_discover_images_reads_split_and_label(dataset: Path) -> None:
    entries = discover_images(dataset)

    assert len(entries) == 11
    assert {(s, label) for _, s, label in entries} >= {
        ("Train", "glioma"),
        ("Test", "pituitary"),
    }


def test_discover_images_fails_on_missing_split(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_images(tmp_path)
