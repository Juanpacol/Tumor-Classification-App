"""The fold assignment of a training run: recorded, shared and independent of the gate."""

import csv
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pytest import MonkeyPatch

from src.tests.training_support import NOW, PER_CLASS, make_config
from src.training import run
from src.training.artifact import MANIFEST_FILE
from src.training.report import FOLDS_FILE
from src.training.run import TrainingResult, train

N_SPLITS = 3


def read_folds(report_dir: Path) -> list[dict[str, str]]:
    with (report_dir / FOLDS_FILE).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validation_rows(rows: list[dict[str, str]]) -> list[list[int]]:
    """Matrix rows of each fold: the used images, in fold-record order."""
    used = [row for row in rows if row["used"] == "1"]
    return [
        [i for i, row in enumerate(used) if row["fold"] == str(fold)]
        for fold in range(N_SPLITS)
    ]


def test_the_fold_record_lists_every_training_image(result: TrainingResult) -> None:
    rows = read_folds(result.report_dir)

    assert len(rows) == 4 * PER_CLASS
    assert all(row["path"].startswith("Train/") for row in rows)
    assert {row["fold"] for row in rows} == {str(f) for f in range(N_SPLITS)}


def test_a_group_of_near_duplicates_stays_in_one_fold(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    lines = workspace["manifest"].read_text().splitlines()
    paired = [lines[0]]
    for line in lines[1:]:
        cells = line.split(",")
        if cells[1] == "Train":
            cells[3] = str(int(cells[3]) // 2)
        paired.append(",".join(cells))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(paired) + "\n")

    trained = train(
        make_config({**workspace, "manifest": manifest}, tmp_path / "out"), now=NOW
    )

    folds_of_group: dict[str, set[str]] = {}
    for row in read_folds(trained.report_dir):
        folds_of_group.setdefault(row["group_id"], set()).add(row["fold"])
    assert len(folds_of_group) == 2 * PER_CLASS
    assert all(len(folds) == 1 for folds in folds_of_group.values())


def test_the_hybrid_ablation_is_the_selection_score(result: TrainingResult) -> None:
    selection = result.report["selection"]

    assert result.report["ablation_macro_f1"]["hybrid"] == (
        selection["scores"][selection["best"]]["mean_macro_f1"]
    )


def test_the_report_counts_used_images_per_fold_and_class(
    result: TrainingResult,
) -> None:
    validation = result.report["validation"]
    rows = read_folds(result.report_dir)

    assert validation["method"] == "StratifiedGroupKFold"
    assert validation["n_splits"] == N_SPLITS and validation["seed"] == 42
    counts = validation["used_per_fold_and_class"]
    assert sum(sum(c.values()) for c in counts.values()) == len(rows)
    for fold, per_class in counts.items():
        for label, count in per_class.items():
            assert count == sum(
                1 for r in rows if r["fold"] == fold and r["label"] == label
            )
    assert "## Folds" in (result.report_dir / "report.md").read_text()


def test_balanced_accuracy_is_reported_and_stored(result: TrainingResult) -> None:
    manifest = json.loads((result.artifact_dir / MANIFEST_FILE).read_text())

    assert 0.0 <= result.report["test"]["balanced_accuracy"] <= 1.0
    assert all(
        "mean_balanced_accuracy" in s
        for s in result.report["selection"]["scores"].values()
    )
    assert manifest["metrics"]["test"]["balanced_accuracy"] == (
        result.report["test"]["balanced_accuracy"]
    )
    assert manifest["calibration"]["ensemble"] is True


def test_selection_ablation_and_calibration_share_the_recorded_folds(
    workspace: dict[str, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def recording(name: str) -> None:
        real = getattr(run, name)
        signature = inspect.signature(real)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            seen[name] = signature.bind(*args, **kwargs).arguments["splits"]
            return real(*args, **kwargs)

        monkeypatch.setattr(run, name, wrapper)

    for name in ("compare_classifiers", "ablate_blocks", "calibrate"):
        recording(name)

    trained = train(make_config(workspace, tmp_path), now=NOW)

    expected = validation_rows(read_folds(trained.report_dir))
    for name, splits in seen.items():
        assert [v.tolist() for _, v in splits] == expected, name


def test_an_image_rejected_by_the_gate_keeps_its_fold_and_is_never_a_row(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    rejected = "Train/glioma/blank.png"
    Image.new("L", (128, 128), 0).save(workspace["data"] / rejected)
    manifest = tmp_path / "manifest.csv"
    lines = workspace["manifest"].read_text().splitlines()
    manifest.write_text(
        "\n".join([*lines, f"{rejected},Train,glioma,99999,1,1,kept"]) + "\n"
    )

    try:
        trained = train(
            make_config({**workspace, "manifest": manifest}, tmp_path / "out"), now=NOW
        )
    finally:
        (workspace["data"] / rejected).unlink()

    rows = read_folds(trained.report_dir)
    blank = [row for row in rows if row["path"] == rejected]
    assert len(blank) == 1 and blank[0]["used"] == "0"
    assert blank[0]["fold"] in {str(f) for f in range(N_SPLITS)}
    assert trained.report["dataset"]["skipped_train"] == 1
    assert sum(len(fold) for fold in validation_rows(rows)) == 4 * PER_CLASS


def test_a_fold_left_without_images_stops_the_run_before_fitting(
    workspace: dict[str, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    fitted: list[int] = []
    monkeypatch.setattr(
        run,
        "assign_folds",
        lambda samples, n_splits, seed: {s.path: 0 for s in samples},
    )
    monkeypatch.setattr(run, "compare_classifiers", lambda *a: fitted.append(1))

    with pytest.raises(ValueError, match=r"folds without usable images: \[1, 2\]"):
        train(make_config(workspace, tmp_path), now=NOW)

    assert fitted == []


def test_the_same_seed_gives_the_same_fold_record(
    workspace: dict[str, Path], result: TrainingResult, tmp_path: Path
) -> None:
    again = train(make_config(workspace, tmp_path), now=NOW)

    assert read_folds(again.report_dir) == read_folds(result.report_dir)
