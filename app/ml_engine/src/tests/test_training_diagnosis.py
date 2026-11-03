from pathlib import Path
from typing import Any

import pytest

from src.contracts import TumorLabel
from src.tests.synthetic import save, synthetic_slice
from src.training.dataset import Sample
from src.training.diagnosis import error_breakdown, predictions_table

LABELS = [label.value for label in TumorLabel]


def row(label: str, predicted: str, **values: Any) -> dict[str, Any]:
    return {
        "label": label,
        "predicted": predicted,
        "width": 256,
        "height": 256,
        "group_size": 1,
        "snr": 5.0,
        "entropy": 6.0,
        **values,
    }


ROWS = [
    row("glioma", "no_tumor", width=512, height=512, snr=1.0),
    row("glioma", "glioma", group_size=3, snr=2.0),
    row("meningioma", "no_tumor", snr=3.0, entropy=1.0),
    row("meningioma", "meningioma", snr=4.0),
    row("no_tumor", "no_tumor", width=512, height=512, snr=5.0),
    row("pituitary", "pituitary", group_size=2, snr=6.0),
]


def test_every_split_partitions_the_images_and_the_errors() -> None:
    breakdown = error_breakdown(ROWS, LABELS, "no_tumor")

    assert set(breakdown["splits"]) == {
        "source_size",
        "group_size",
        "snr_tercile",
        "entropy_tercile",
    }
    for cells in breakdown["splits"].values():
        assert sum(c["images"] for c in cells.values()) == len(ROWS)
        assert sum(c["tumor_as_no_tumor"] for c in cells.values()) == 2


def test_cells_count_recall_and_tumors_called_no_tumor() -> None:
    cells = error_breakdown(ROWS, LABELS, "no_tumor")["splits"]["source_size"]

    assert cells["512x512"] == {
        "images": 2,
        "recall": {
            "glioma": 0.0,
            "meningioma": None,
            "pituitary": None,
            "no_tumor": 1.0,
        },
        "tumor_as_no_tumor": 1,
    }
    assert cells["other"]["recall"]["meningioma"] == 0.5


def test_tercile_edges_come_from_the_rows() -> None:
    breakdown = error_breakdown(ROWS, LABELS, "no_tumor")

    low, high = breakdown["tercile_edges"]["snr"]
    assert low == pytest.approx(8 / 3) and high == pytest.approx(13 / 3)
    terciles = breakdown["splits"]["snr_tercile"]
    assert [terciles[t]["images"] for t in ("low", "mid", "high")] == [2, 2, 2]


def test_the_table_has_one_row_per_image_with_its_metadata(tmp_path: Path) -> None:
    data = tmp_path / "data"
    samples = [
        Sample(save(synthetic_slice(i, size=64), data / "Train" / f"{i}.png"),
               "Train", TumorLabel.GLIOMA, group)
        for i, group in enumerate([7, 7, 9])
    ]  # fmt: skip

    rows = predictions_table(
        samples,
        data,
        [0, 1, 2],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        ["glioma", "no_tumor", "glioma"],
    )

    assert [r["path"] for r in rows] == ["Train/0.png", "Train/1.png", "Train/2.png"]
    assert [r["group_size"] for r in rows] == [2, 2, 1]
    assert rows[1] == {
        "path": "Train/1.png",
        "label": "glioma",
        "fold": 1,
        "group_id": 7,
        "group_size": 2,
        "width": 64,
        "height": 64,
        "snr": 3.0,
        "entropy": 4.0,
        "predicted": "no_tumor",
    }
