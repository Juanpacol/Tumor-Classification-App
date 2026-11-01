import csv
from pathlib import Path
from typing import Any

import pytest

from src.training.folds import FoldRow
from src.training.report import (
    FOLDS_FILE,
    TARGET_MACRO_F1,
    build_report,
    render_markdown,
    write_report,
)

CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]


def metrics_dict(macro_f1: float) -> dict[str, Any]:
    return {
        "accuracy": 0.9,
        "macro_f1": macro_f1,
        "balanced_accuracy": 0.88,
        "per_class": {
            name: {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 10}
            for name in CLASSES
        },
        "confusion_matrix": [[10 if i == j else 0 for j in range(4)] for i in range(4)],
    }


def make_report(macro_f1: float) -> dict[str, Any]:
    return build_report(
        created_at="2026-10-29T10:00:00+00:00",
        version="1.0.0",
        seed=42,
        dataset={"train_images": 100, "test_images": 40},
        validation={
            "method": "StratifiedGroupKFold",
            "group": "group_id",
            "n_splits": 2,
            "seed": 42,
            "used_per_fold_and_class": {
                "0": {"glioma": 3, "no_tumor": 1},
                "1": {"glioma": 2, "no_tumor": 2},
            },
        },
        selection={
            "scores": {
                "svm_rbf": {
                    "mean_macro_f1": 0.9,
                    "std_macro_f1": 0.02,
                    "mean_balanced_accuracy": 0.87,
                    "std_balanced_accuracy": 0.03,
                }
            },
            "best": "svm_rbf",
        },
        ablation={"cnn_only": 0.8, "radiomics_only": 0.6, "hybrid": 0.9},
        augmentation={
            "on": False,
            "copies": 1,
            "ranges": {"max_rotation_deg": 10.0},
            "cv_without": {"mean": 0.9, "std": 0.02},
            "cv_with": {"mean": 0.91, "std": 0.03},
            "margin": 0.02,
        },
        test=metrics_dict(macro_f1),
        calibration_method="sigmoid",
        shortcut_audit=shortcut_audit(0.55),
        guarantees=[
            {
                "id": "G1",
                "guarantee": "No shared group_id",
                "status": "checked in this run",
            }
        ],
    )


def shortcut_audit(balanced_accuracy: float) -> dict[str, Any]:
    return {
        "metadata_baseline": {
            "features": ["width", "height", "file_size"],
            "classifier": "random_forest",
            "mean_balanced_accuracy": balanced_accuracy,
            "std_balanced_accuracy": 0.04,
            "warning_level": 0.7,
            "warning": balanced_accuracy >= 0.7,
        },
        "no_tumor_test_recall_by_source_size": {
            "512x512": {"recall": None, "support": 0},
            "other": {"recall": 0.875, "support": 8},
        },
    }


@pytest.mark.parametrize(
    ("macro_f1", "met"),
    [
        (TARGET_MACRO_F1, True),
        (TARGET_MACRO_F1 + 0.05, True),
        (TARGET_MACRO_F1 - 0.001, False),
    ],
)
def test_target_is_met_at_or_above_the_threshold(macro_f1: float, met: bool) -> None:
    assert make_report(macro_f1)["target_met"] is met


def test_markdown_states_the_verdict_and_shows_every_section() -> None:
    markdown = render_markdown(make_report(0.95))

    assert "Test macro-F1 0.950" in markdown and "Target 0.90: met" in markdown
    assert "balanced accuracy 0.880" in markdown
    for heading in (
        "Folds",
        "Per class",
        "Confusion matrix",
        "Classifier selection",
        "Ablation",
        "Augmentation",
        "Shortcut audit",
        "Leakage guarantees",
        "Data",
    ):
        assert heading in markdown
    assert "| glioma |" in markdown and "svm_rbf" in markdown and "hybrid" in markdown


def test_markdown_shows_fold_counts_and_cv_balanced_accuracy() -> None:
    markdown = render_markdown(make_report(0.95))

    assert "| Fold | glioma | no_tumor | Total |" in markdown
    assert "| 0 | 3 | 1 | 4 |" in markdown and "| 1 | 2 | 2 | 4 |" in markdown
    assert "| svm_rbf | 0.900 | 0.020 | 0.870 | 0.030 |" in markdown


def test_markdown_lists_per_class_counts_inline() -> None:
    report = make_report(0.95)
    report["dataset"]["split_removed_per_class"] = {"glioma": 142, "notumor": 15}

    markdown = render_markdown(report)

    assert "- split_removed_per_class: glioma 142, notumor 15" in markdown


def test_markdown_flags_a_missed_target() -> None:
    assert "NOT met" in render_markdown(make_report(0.5))


FOLDS = [
    FoldRow("Train/glioma/a.png", "glioma", 7, 0, True),
    FoldRow("Train/glioma/b.png", "glioma", 7, 0, False),
]


def test_report_is_written_as_json_markdown_and_fold_record(tmp_path: Path) -> None:
    write_report(tmp_path / "20261029T100000Z", make_report(0.95), FOLDS)

    folder = tmp_path / "20261029T100000Z"
    assert (folder / "report.json").is_file()
    assert (folder / "report.md").is_file()
    with (folder / FOLDS_FILE).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["path"], r["group_id"], r["fold"], r["used"]) for r in rows] == [
        ("Train/glioma/a.png", "7", "0", "1"),
        ("Train/glioma/b.png", "7", "0", "0"),
    ]


def test_an_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    write_report(tmp_path / "run", make_report(0.95), FOLDS)

    with pytest.raises(FileExistsError):
        write_report(tmp_path / "run", make_report(0.95), FOLDS)


def test_markdown_shows_the_augmentation_decision() -> None:
    markdown = render_markdown(make_report(0.95))

    assert "| originals | 0.900 | 0.020 |" in markdown
    assert "| originals + 1 copy per image | 0.910 | 0.030 |" in markdown
    assert "Augmentation **off**" in markdown and "at least 0.020" in markdown


def test_markdown_counts_several_copies_in_the_plural() -> None:
    report = make_report(0.95)
    report["augmentation"] = {**report["augmentation"], "copies": 2}

    assert "| originals + 2 copies per image |" in render_markdown(report)


def test_markdown_shows_the_shortcut_audit_and_the_guarantees() -> None:
    markdown = render_markdown(make_report(0.95))

    assert "balanced accuracy of **0.550**" in markdown
    assert "Below the warning level." in markdown
    assert "| 512x512 | n/a | 0 |" in markdown and "| other | 0.875 | 8 |" in markdown
    assert "| G1 | No shared group_id | checked in this run |" in markdown


def test_markdown_flags_a_metadata_shortcut() -> None:
    report = make_report(0.95)
    report["shortcut_audit"] = shortcut_audit(0.8)

    assert "WARNING: at or above the warning level" in render_markdown(report)
