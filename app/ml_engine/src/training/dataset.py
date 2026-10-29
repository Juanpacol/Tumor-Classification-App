"""Training samples from the dataset folder and the cleaning manifest."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.contracts import TumorLabel

FOLDER_TO_LABEL = {
    "glioma": TumorLabel.GLIOMA,
    "meningioma": TumorLabel.MENINGIOMA,
    "notumor": TumorLabel.NO_TUMOR,
    "pituitary": TumorLabel.PITUITARY,
}
SPLITS = ("Train", "Test")
_COLUMNS = {"path", "split", "label", "group_id", "keep"}


@dataclass(frozen=True)
class Sample:
    """One kept image; ``group_id`` joins near-duplicate images of the same scan group."""

    path: Path
    split: str
    label: TumorLabel
    group_id: int


def load_samples(data_dir: Path, manifest_path: Path) -> list[Sample]:
    """Kept images listed in the ``manifest.csv`` written by the deduplication step.

    Paths in the manifest are relative to ``data_dir``. Raises ``ValueError`` for a
    manifest with missing columns, an unknown label or an unknown split, and
    ``FileNotFoundError`` for a listed image that is not on disk.
    """
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = _COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        rows = list(reader)

    samples = []
    for row in rows:
        if row["keep"] != "1":
            continue
        if row["label"] not in FOLDER_TO_LABEL:
            raise ValueError(f"unknown label {row['label']!r} in manifest")
        if row["split"] not in SPLITS:
            raise ValueError(f"unknown split {row['split']!r} in manifest")
        path = data_dir / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"image listed in the manifest not found: {path}")
        samples.append(
            Sample(
                path, row["split"], FOLDER_TO_LABEL[row["label"]], int(row["group_id"])
            )
        )
    return samples


def split_samples(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    """``(train, test)`` according to the official folders."""
    return (
        [s for s in samples if s.split == "Train"],
        [s for s in samples if s.split == "Test"],
    )
