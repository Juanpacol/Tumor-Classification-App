from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.contracts import TumorLabel
from src.training.dataset import Sample
from src.training.folds import fold_splits
from src.training.shortcut import (
    WARNING_BALANCED_ACCURACY,
    metadata_baseline,
    metadata_matrix,
    recall_by_source_size,
)

NO_TUMOR = TumorLabel.NO_TUMOR.value


def image_sample(
    path: Path, size: tuple[int, int], label: TumorLabel, group: int
) -> Sample:
    Image.new("L", size, color=group % 255).save(path)
    return Sample(path, "Train", label, group)


def test_metadata_is_width_height_and_file_size(tmp_path: Path) -> None:
    sample = image_sample(tmp_path / "a.png", (40, 30), TumorLabel.GLIOMA, 0)

    [row] = metadata_matrix([sample])

    assert row.tolist() == [40, 30, sample.path.stat().st_size]


def size_dataset(tmp_path: Path, size_tells_the_class: bool) -> list[Sample]:
    rng = np.random.default_rng(0)
    samples = []
    for index in range(60):
        label = TumorLabel.GLIOMA if index % 2 else TumorLabel.NO_TUMOR
        if size_tells_the_class:
            size = (512, 512) if label is TumorLabel.GLIOMA else (256, 300)
        else:
            width, height = rng.integers(200, 300, size=2)
            size = (int(width), int(height))
        samples.append(image_sample(tmp_path / f"{index}.png", size, label, index))
    return samples


@pytest.mark.parametrize("size_tells_the_class", [True, False])
def test_the_baseline_warns_only_when_the_size_gives_the_class_away(
    tmp_path: Path, size_tells_the_class: bool
) -> None:
    samples = size_dataset(tmp_path, size_tells_the_class)
    y = np.array([s.label.value for s in samples])
    splits = fold_splits(np.arange(len(samples)) % 3, 3)

    baseline = metadata_baseline(samples, y, splits, seed=0)

    assert baseline["warning"] is size_tells_the_class
    assert baseline["warning_level"] == WARNING_BALANCED_ACCURACY
    if size_tells_the_class:
        assert baseline["mean_balanced_accuracy"] == 1.0


def test_no_tumor_recall_is_split_by_source_size(tmp_path: Path) -> None:
    samples = [
        image_sample(tmp_path / "a.png", (512, 512), TumorLabel.NO_TUMOR, 0),
        image_sample(tmp_path / "b.png", (300, 280), TumorLabel.NO_TUMOR, 1),
        image_sample(tmp_path / "c.png", (300, 280), TumorLabel.NO_TUMOR, 2),
        image_sample(tmp_path / "d.png", (512, 512), TumorLabel.GLIOMA, 3),
    ]
    y = np.array([s.label.value for s in samples])
    predictions = np.array([NO_TUMOR, NO_TUMOR, "glioma", NO_TUMOR])

    by_size = recall_by_source_size(samples, y, predictions, NO_TUMOR)

    assert by_size == {
        "512x512": {"recall": 1.0, "support": 1},
        "other": {"recall": 0.5, "support": 2},
    }


def test_a_size_group_without_the_label_has_no_recall(tmp_path: Path) -> None:
    samples = [image_sample(tmp_path / "a.png", (300, 280), TumorLabel.NO_TUMOR, 0)]

    by_size = recall_by_source_size(
        samples, np.array([NO_TUMOR]), np.array([NO_TUMOR]), NO_TUMOR
    )

    assert by_size["512x512"] == {"recall": None, "support": 0}
