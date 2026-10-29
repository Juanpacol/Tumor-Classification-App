from pathlib import Path
from typing import Any

import pytest

from src.training.report import (
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
        selection={
            "scores": {"svm_rbf": {"mean_macro_f1": 0.9, "std_macro_f1": 0.02}},
            "best": "svm_rbf",
        },
        ablation={"cnn_only": 0.8, "radiomics_only": 0.6, "hybrid": 0.9},
        test=metrics_dict(macro_f1),
        calibration_method="sigmoid",
    )


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
    for heading in (
        "Per class",
        "Confusion matrix",
        "Classifier selection",
        "Ablation",
        "Data",
    ):
        assert heading in markdown
    assert "| glioma |" in markdown and "svm_rbf" in markdown and "hybrid" in markdown


def test_markdown_flags_a_missed_target() -> None:
    assert "NOT met" in render_markdown(make_report(0.5))


def test_report_is_written_as_json_and_markdown(tmp_path: Path) -> None:
    write_report(tmp_path / "20261029T100000Z", make_report(0.95))

    assert (tmp_path / "20261029T100000Z" / "report.json").is_file()
    assert (tmp_path / "20261029T100000Z" / "report.md").is_file()


def test_an_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    write_report(tmp_path / "run", make_report(0.95))

    with pytest.raises(FileExistsError):
        write_report(tmp_path / "run", make_report(0.95))
