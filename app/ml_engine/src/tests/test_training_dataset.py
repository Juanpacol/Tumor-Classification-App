import csv
from pathlib import Path

import pytest

from src.contracts import TumorLabel
from src.tests.synthetic import save, synthetic_slice
from src.training.dataset import Sample, load_samples, split_samples

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


def sample(name: str, split: str, label: TumorLabel, group_id: int) -> Sample:
    return Sample(Path(name), split, label, group_id)


def test_train_image_sharing_a_group_with_test_is_removed() -> None:
    shared = sample("tr.png", "Train", TumorLabel.GLIOMA, 7)
    twin = sample("te.png", "Test", TumorLabel.GLIOMA, 7)

    split = split_samples([shared, twin])

    assert split.train == []
    assert split.removed_per_class[TumorLabel.GLIOMA] == 1


def test_train_image_without_test_group_member_is_kept() -> None:
    alone = sample("tr.png", "Train", TumorLabel.MENINGIOMA, 1)
    other = sample("te.png", "Test", TumorLabel.MENINGIOMA, 2)

    split = split_samples([alone, other])

    assert split.train == [alone]


def test_group_shared_only_with_a_dropped_test_image_keeps_the_train_image(
    data_dir: Path, tmp_path: Path
) -> None:
    manifest = write_manifest(
        tmp_path / "manifest.csv",
        [
            ["Train/glioma/a.png", "Train", "glioma", 5, 1, 1, "kept"],
            ["Test/glioma/gone.png", "Test", "glioma", 5, 1, 0, "duplicate_of:x"],
            ["Test/pituitary/c.png", "Test", "pituitary", 6, 1, 1, "kept"],
        ],
    )

    split = split_samples(load_samples(data_dir, manifest))

    assert [s.path.name for s in split.train] == ["a.png"]
    assert sum(split.removed_per_class.values()) == 0


def test_test_set_is_returned_unchanged() -> None:
    samples = [
        sample("te1.png", "Test", TumorLabel.PITUITARY, 3),
        sample("tr.png", "Train", TumorLabel.PITUITARY, 3),
        sample("te2.png", "Test", TumorLabel.NO_TUMOR, 4),
    ]

    split = split_samples(samples)

    assert split.test == [samples[0], samples[2]]


def test_removed_counts_list_every_class() -> None:
    split = split_samples(
        [
            sample("tr.png", "Train", TumorLabel.GLIOMA, 0),
            sample("te.png", "Test", TumorLabel.GLIOMA, 0),
        ]
    )

    assert split.removed_per_class == {
        TumorLabel.GLIOMA: 1,
        TumorLabel.MENINGIOMA: 0,
        TumorLabel.NO_TUMOR: 0,
        TumorLabel.PITUITARY: 0,
    }


def test_no_group_is_in_both_splits() -> None:
    labels = list(TumorLabel)
    samples = [
        sample(f"{split}{i}.png", split, labels[i % 4], i % 7)
        for split in ("Train", "Test")
        for i in range(30 if split == "Train" else 9)
    ]

    split = split_samples(samples)

    assert not {s.group_id for s in split.train} & {s.group_id for s in split.test}
    assert len(split.train) + sum(split.removed_per_class.values()) == 30


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
