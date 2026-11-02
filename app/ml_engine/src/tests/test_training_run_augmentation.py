"""Augmented copies in a training run: where they go and how the run decides on them."""

import csv
import inspect
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pytest import MonkeyPatch

from src.deep_learning.bundle import sha256_of
from src.tests.training_support import NOW, PER_CLASS, make_config
from src.training import run
from src.training.artifact import MANIFEST_FILE
from src.training.model import CandidateScore, SelectionResult
from src.training.report import FOLDS_FILE
from src.training.run import TrainingResult, train

N_TRAIN = 4 * PER_CLASS


def score(mean: float, std: float) -> CandidateScore:
    return CandidateScore(mean, std, (mean,), mean, std, (mean,))


def copy_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("*/aug-*/*/*.npz"))


def test_copies_are_made_only_of_usable_training_images(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    rejected = "Train/glioma/blank_copy.png"
    Image.new("L", (128, 128), 0).save(workspace["data"] / rejected)
    manifest = tmp_path / "manifest.csv"
    lines = workspace["manifest"].read_text().splitlines()
    manifest.write_text(
        "\n".join([*lines, f"{rejected},Train,glioma,99998,1,1,kept"]) + "\n"
    )
    config = replace(
        make_config({**workspace, "manifest": manifest}, tmp_path / "out"),
        cache_dir=tmp_path / "cache",
    )

    try:
        trained = train(config, now=NOW)
        blank_digest = sha256_of(workspace["data"] / rejected)
    finally:
        (workspace["data"] / rejected).unlink()

    with (trained.report_dir / FOLDS_FILE).open(newline="", encoding="utf-8") as f:
        used = [row["path"] for row in csv.DictReader(f) if row["used"] == "1"]
    test_paths = [line.split(",")[0] for line in lines[1:] if ",Test," in line]
    digests = {path.stem for path in copy_files(tmp_path / "cache")}
    assert len(digests) == len(used) == N_TRAIN
    assert digests == {sha256_of(workspace["data"] / path) for path in used}
    assert blank_digest not in digests
    assert not digests & {sha256_of(workspace["data"] / path) for path in test_paths}


def test_the_decision_is_in_the_report_and_the_manifest(result: TrainingResult) -> None:
    manifest = json.loads((result.artifact_dir / MANIFEST_FILE).read_text())
    augmentation = result.report["augmentation"]

    assert manifest["augmentation"] == augmentation
    assert augmentation["copies"] == 1
    assert augmentation["ranges"]["max_rotation_deg"] == 10.0
    assert augmentation["margin"] == augmentation["cv_without"]["std"]
    assert augmentation["on"] == (
        augmentation["cv_with"]["mean"]
        >= augmentation["cv_without"]["mean"] + augmentation["cv_without"]["std"]
    )
    assert "## Augmentation" in (result.report_dir / "report.md").read_text()


@pytest.mark.parametrize(("mean_with", "on"), [(0.83, True), (0.81, False)])
def test_augmentation_stays_on_only_above_the_margin(
    workspace: dict[str, Path],
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    mean_with: float,
    on: bool,
) -> None:
    seen: dict[str, Any] = {}
    predictions = np.array(["glioma"] * N_TRAIN)
    monkeypatch.setattr(
        run,
        "compare_classifiers",
        lambda *a, **k: SelectionResult(
            {"logistic_regression": score(0.80, 0.02)},
            "logistic_regression",
            {"logistic_regression": predictions},
        ),
    )
    monkeypatch.setattr(
        run,
        "cross_validate_predictions",
        lambda *a, **k: (score(mean_with, 0.03), predictions),
    )
    for name in ("ablate_blocks", "calibrate"):
        real = getattr(run, name)
        signature = inspect.signature(real)

        def wrapper(
            *args: Any,
            _real: Any = real,
            _name: str = name,
            _signature: inspect.Signature = signature,
            **kwargs: Any,
        ) -> Any:
            arguments = _signature.bind(*args, **kwargs).arguments
            seen[_name] = (len(arguments["matrix"]), arguments["splits"])
            return _real(*args, **kwargs)

        monkeypatch.setattr(run, name, wrapper)

    trained = train(make_config(workspace, tmp_path), now=NOW)

    assert trained.report["augmentation"]["on"] is on
    assert trained.report["ablation_macro_f1"]["hybrid"] == (0.83 if on else 0.80)
    for name, (rows, splits) in seen.items():
        largest_fit_row = max(int(fit.max()) for fit, _ in splits)
        assert rows == (2 * N_TRAIN if on else N_TRAIN), name
        assert (largest_fit_row >= N_TRAIN) is on, name
        assert all(int(validation.max()) < N_TRAIN for _, validation in splits), name


def test_the_serving_pipeline_never_loads_the_augmentation_code() -> None:
    code = (
        "import sys, src.pipelines, src.pipelines.artifact; "
        "print('src.training.augmentation' in sys.modules)"
    )
    root = Path(__file__).resolve().parents[2]

    loaded = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert loaded.stdout.strip() == "False"


def test_no_copy_is_scored_on_testing(
    workspace: dict[str, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    scored: list[int] = []
    real = run.evaluate

    def counting(model: Any, matrix: np.ndarray, *args: Any) -> Any:
        scored.append(len(matrix))
        return real(model, matrix, *args)

    monkeypatch.setattr(run, "evaluate", counting)

    trained = train(make_config(workspace, tmp_path), now=NOW)

    assert scored == [trained.report["dataset"]["test_images"]]
