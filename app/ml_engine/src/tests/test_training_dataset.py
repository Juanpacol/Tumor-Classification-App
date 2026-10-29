import csv
from pathlib import Path

import pytest

from src.contracts import TumorLabel
from src.tests.synthetic import save, synthetic_slice
from src.training.dataset import load_samples, split_samples

HEADER = ["path", "split", "label", "group_id", "group_size", "keep", "reason"]


def write_manifest(
    path: Path, rows: list[list[object]], header: list[str] = HEADER
) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    for relative in (
        "Train/glioma/a.png",
        "Train/notumor/b.png",
        "Test/pituitary/c.png",
    ):
        save(synthetic_slice(1, size=64), tmp_path / "data" / relative)
    return tmp_path / "data"


def test_kept_images_are_loaded_with_mapped_labels(
    data_dir: Path, tmp_path: Path
) -> None:
    manifest = write_manifest(
        tmp_path / "manifest.csv",
        [
            ["Train/glioma/a.png", "Train", "glioma", 0, 1, 1, "kept"],
            ["Train/notumor/b.png", "Train", "notumor", 1, 1, 1, "kept"],
            ["Test/pituitary/c.png", "Test", "pituitary", 2, 1, 1, "kept"],
        ],
    )

    samples = load_samples(data_dir, manifest)

    assert [s.label for s in samples] == [
        TumorLabel.GLIOMA,
        TumorLabel.NO_TUMOR,
        TumorLabel.PITUITARY,
    ]
    assert samples[1].group_id == 1
    assert samples[0].path == data_dir / "Train/glioma/a.png"


def test_removed_images_are_skipped_even_if_missing_on_disk(
    data_dir: Path, tmp_path: Path
) -> None:
    manifest = write_manifest(
        tmp_path / "manifest.csv",
        [
            ["Train/glioma/a.png", "Train", "glioma", 0, 1, 1, "kept"],
            ["Train/glioma/gone.png", "Train", "glioma", 0, 1, 0, "duplicate_of:x"],
        ],
    )

    assert len(load_samples(data_dir, manifest)) == 1


def test_split_samples_follows_the_official_folders(
    data_dir: Path, tmp_path: Path
) -> None:
    manifest = write_manifest(
        tmp_path / "manifest.csv",
        [
            ["Train/glioma/a.png", "Train", "glioma", 0, 1, 1, "kept"],
            ["Test/pituitary/c.png", "Test", "pituitary", 2, 1, 1, "kept"],
        ],
    )

    train, test = split_samples(load_samples(data_dir, manifest))

    assert [s.split for s in train] == ["Train"] and [s.split for s in test] == ["Test"]


def test_missing_columns_are_rejected(data_dir: Path, tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path / "manifest.csv", [], header=["path", "label"])

    with pytest.raises(ValueError, match="missing columns"):
        load_samples(data_dir, manifest)


@pytest.mark.parametrize(
    ("label", "split", "message"),
    [("lymphoma", "Train", "unknown label"), ("glioma", "Val", "unknown split")],
)
def test_unknown_label_or_split_is_rejected(
    data_dir: Path, tmp_path: Path, label: str, split: str, message: str
) -> None:
    manifest = write_manifest(
        tmp_path / "manifest.csv",
        [["Train/glioma/a.png", split, label, 0, 1, 1, "kept"]],
    )

    with pytest.raises(ValueError, match=message):
        load_samples(data_dir, manifest)


def test_listed_image_missing_on_disk_is_an_error(
    data_dir: Path, tmp_path: Path
) -> None:
    manifest = write_manifest(
        tmp_path / "manifest.csv",
        [["Train/glioma/nope.png", "Train", "glioma", 0, 1, 1, "kept"]],
    )

    with pytest.raises(FileNotFoundError):
        load_samples(data_dir, manifest)
