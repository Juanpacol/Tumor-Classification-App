from pathlib import Path

import pytest
from PIL import Image

from src.preprocessing.deduplication.discovery import discover_images
from src.preprocessing.deduplication.fingerprint import (
    ImageFingerprint,
    compute_fingerprint,
)
from src.tests.synthetic import save, synthetic_slice


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """Small dataset with known duplicates.

    * ``Train/glioma/a.png`` + byte copy ``Test/glioma/a_copy.png``       -> exact duplicate
    * ``Train/glioma/b.png`` + ``Test/glioma/b_small.jpg`` (resized, JPEG) -> near duplicate
    * ``Train/glioma/c.png`` + ``Train/glioma/c_flip.png`` (mirrored)      -> near duplicate
    * ``Train/meningioma/Tr-aug-me_1.png`` (brightened copy of ``d.png``)  -> augmented
    * ``Train/meningioma/e.png`` + ``Test/pituitary/e_other.png``          -> label conflict
    * ``Test/meningioma/f.png``                                           -> unique
    """
    data_dir = tmp_path / "data"
    a, b, c, d, e, f = (synthetic_slice(seed) for seed in range(6))

    a_path = save(a, data_dir / "Train/glioma/a.png")
    (data_dir / "Test/glioma").mkdir(parents=True)
    (data_dir / "Test/glioma/a_copy.png").write_bytes(a_path.read_bytes())
    save(b, data_dir / "Train/glioma/b.png")
    save(
        b.resize((180, 180), Image.Resampling.BILINEAR),
        data_dir / "Test/glioma/b_small.jpg",
        quality=80,
    )
    save(c, data_dir / "Train/glioma/c.png")
    save(
        c.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        data_dir / "Train/glioma/c_flip.png",
    )
    save(d, data_dir / "Train/meningioma/d.png")
    save(
        d.point(lambda v: min(255, int(v * 1.05))),
        data_dir / "Train/meningioma/Tr-aug-me_1.png",
    )
    save(e, data_dir / "Train/meningioma/e.png")
    save(e, data_dir / "Test/pituitary/e_other.png")
    save(f, data_dir / "Test/meningioma/f.png")
    return data_dir


@pytest.fixture()
def fingerprints(dataset: Path) -> list[ImageFingerprint]:
    """Fingerprints of every image in ``dataset``."""
    return [compute_fingerprint(*entry) for entry in discover_images(dataset)]


def index_of(fingerprints: list[ImageFingerprint], name: str) -> int:
    """Position of the image with file name ``name``."""
    return [fp.path.name for fp in fingerprints].index(name)
