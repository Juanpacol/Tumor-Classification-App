import json
from pathlib import Path
from typing import Any

import joblib
import pytest
from pytest import CaptureFixture, MonkeyPatch

from src.contracts import TumorLabel
from src.tests.training_support import NOW, PER_CLASS, make_config
from src.training import __main__ as cli
from src.training import run
from src.training.artifact import CLASSIFIER_FILE, MANIFEST_FILE
from src.training.config import apply_round
from src.training.model import candidate_classifiers
from src.training.report import CONFIG_FILE
from src.training.run import TrainingResult, train
from src.training.shortcut import WARNING_BALANCED_ACCURACY


def test_a_run_writes_the_artifact_and_a_timestamped_report(
    result: TrainingResult,
) -> None:
    assert result.artifact_dir.name == "model-v9.9.9"
    assert (result.artifact_dir / CLASSIFIER_FILE).is_file()
    assert (result.artifact_dir / MANIFEST_FILE).is_file()
    assert result.report_dir.name == "20261029T100000Z"
    assert (result.report_dir / "report.json").is_file()
    assert (result.report_dir / "report.md").is_file()


def test_the_manifest_records_how_the_model_was_trained(result: TrainingResult) -> None:
    manifest = json.loads((result.artifact_dir / MANIFEST_FILE).read_text())

    assert manifest["version"] == "9.9.9" and manifest["seed"] == 42
    assert manifest["classifier"] in candidate_classifiers()
    assert manifest["calibration"]["method"] == "sigmoid"
    assert manifest["preprocessing"]["cnn_size"] == 224
    assert manifest["preprocessing"]["crop_margin"] == 0.05
    assert manifest["quality_thresholds"] == {
        "min_snr": 2.74,
        "min_entropy": 3.2,
        "metrics_version": 2,
    }
    assert len(manifest["radiomic_feature_names"]) == 45
    assert manifest["dataset"]["split_removed_per_class"] == {
        label.value: 0 for label in TumorLabel
    }
    assert manifest["metrics"]["test"]["macro_f1"] == result.report["test"]["macro_f1"]


def test_the_report_has_metrics_selection_and_ablation(result: TrainingResult) -> None:
    report = result.report

    assert report["dataset"]["train_images"] == 4 * PER_CLASS
    assert report["dataset"]["skipped_train"] == 0
    assert set(report["dataset"]["split_removed_per_class"]) == {
        label.value for label in TumorLabel
    }
    assert set(report["selection"]["scores"]) == set(candidate_classifiers())
    assert set(report["ablation_macro_f1"]) == {"cnn_only", "radiomics_only", "hybrid"}
    assert len(report["test"]["confusion_matrix"]) == 4
    assert report["test"]["macro_f1"] > 0.5


def test_the_saved_classifier_gives_calibrated_probabilities(
    result: TrainingResult, workspace: dict[str, Path]
) -> None:
    model = joblib.load(result.artifact_dir / CLASSIFIER_FILE)

    assert len(model.classes_) == 4


def test_the_test_split_is_scored_exactly_once(
    workspace: dict[str, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    calls = []
    real = run.evaluate
    monkeypatch.setattr(
        run, "evaluate", lambda *a, **k: calls.append(1) or real(*a, **k)
    )

    train(make_config(workspace, tmp_path), now=NOW)

    assert len(calls) == 1


def test_the_same_seed_reproduces_the_same_metrics(
    workspace: dict[str, Path], result: TrainingResult, tmp_path: Path
) -> None:
    again = train(make_config(workspace, tmp_path), now=NOW)

    assert again.report["test"] == result.report["test"]
    assert again.report["selection"] == result.report["selection"]
    assert again.report["ablation_macro_f1"] == result.report["ablation_macro_f1"]


def test_an_existing_artifact_stops_the_run_before_any_work(
    workspace: dict[str, Path], result: TrainingResult, tmp_path: Path
) -> None:
    config = make_config(workspace, result.artifact_dir.parent)

    with pytest.raises(FileExistsError):
        train(config, now=NOW)


def test_a_manifest_without_training_images_is_an_error(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    lines = workspace["manifest"].read_text().splitlines()
    only_test = [lines[0]] + [line for line in lines[1:] if line.startswith("Test/")]
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(only_test) + "\n")
    config = make_config({**workspace, "manifest": manifest}, tmp_path / "out")

    with pytest.raises(ValueError, match="no usable training images"):
        train(config, now=NOW)


def test_the_cli_prints_where_the_results_went(
    workspace: dict[str, Path], tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    cli.main(
        [
            "--data-dir", str(workspace["data"]),
            "--manifest", str(workspace["manifest"]),
            "--backbone-bundle", str(workspace["backbone"]),
            "--out", str(tmp_path),
            "--n-splits", "3",
            "--version", "cli-test",
            "--cache-dir", str(workspace["root"] / "cache"),
        ]
    )  # fmt: skip

    output = capsys.readouterr().out
    assert "model-vcli-test" in output and "test macro-F1" in output
    assert "removed from Train (group in Test): glioma 0" in output


def cli_report(target_met: bool, baseline: float) -> dict[str, Any]:
    return {
        "test": {"macro_f1": 0.5, "accuracy": 0.5},
        "target_met": target_met,
        "target_macro_f1": 0.9,
        "dataset": {"split_removed_per_class": {"glioma": 0}},
        "shortcut_audit": {
            "metadata_baseline": {
                "mean_balanced_accuracy": baseline,
                "warning_level": WARNING_BALANCED_ACCURACY,
                "warning": baseline >= WARNING_BALANCED_ACCURACY,
            }
        },
    }


def run_cli_with(
    report: dict[str, Any], monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli, "train", lambda config: TrainingResult(tmp_path, tmp_path, report)
    )
    cli.main(["--data-dir", "d", "--manifest", "m"])


def test_the_cli_warns_when_the_target_is_missed(
    capsys: CaptureFixture[str], monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    run_cli_with(cli_report(False, 0.3), monkeypatch, tmp_path)

    assert "not met" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("baseline", "warned"),
    [(WARNING_BALANCED_ACCURACY, True), (WARNING_BALANCED_ACCURACY - 0.01, False)],
)
def test_the_cli_warns_when_metadata_alone_predicts_the_class(
    baseline: float,
    warned: bool,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_cli_with(cli_report(True, baseline), monkeypatch, tmp_path)

    output = capsys.readouterr()
    assert f"metadata-only baseline balanced accuracy {baseline:.3f}" in output.out
    assert ("Present the result to the team" in output.err) is warned


def test_the_report_has_the_shortcut_audit_and_guarantees(
    result: TrainingResult,
) -> None:
    audit = result.report["shortcut_audit"]

    assert audit["metadata_baseline"]["features"] == ["width", "height", "file_size"]
    assert 0.0 <= audit["metadata_baseline"]["mean_balanced_accuracy"] <= 1.0
    by_size = audit["no_tumor_test_recall_by_source_size"]
    assert by_size["512x512"]["support"] == 0
    assert by_size["other"]["support"] == PER_CLASS
    ids = [g["id"] for g in result.report["leakage_guarantees"]]
    assert ids == ["G1", "G2", "G3", "G7", "G8"]


def test_the_report_has_out_of_fold_metrics_on_train(result: TrainingResult) -> None:
    oof = result.report["cv_out_of_fold"]

    assert isinstance(oof["augmented"], bool)
    assert (
        sum(map(sum, oof["confusion_matrix"]))
        == result.report["dataset"]["train_images"]
    )
    assert {label: m["support"] for label, m in oof["per_class"].items()} == {
        label.value: PER_CLASS for label in TumorLabel
    }
    manifest = json.loads((result.artifact_dir / MANIFEST_FILE).read_text())
    assert manifest["metrics"]["cv_out_of_fold"] == oof


def test_the_run_records_its_resolved_configuration(result: TrainingResult) -> None:
    config = json.loads((result.report_dir / CONFIG_FILE).read_text())
    report = json.loads((result.report_dir / "report.json").read_text())
    manifest = json.loads((result.artifact_dir / MANIFEST_FILE).read_text())

    assert config == report["config"]
    assert manifest["config"] == {
        key: config[key] for key in ("run", "model", "classifiers")
    }
    assert manifest["preprocessing"] == config["preprocessing"]
    assert config["run"] == {"seed": 42, "n_splits": 3}
    assert result.report["config_source"] is None is manifest["config_source"]


def npz_files(cache: Path) -> dict[Path, int]:
    return {path: path.stat().st_mtime_ns for path in cache.rglob("*.npz")}


def test_a_classifier_only_round_reuses_every_cached_feature(
    workspace: dict[str, Path], result: TrainingResult, tmp_path: Path
) -> None:
    round_file = tmp_path / "round.toml"
    round_file.write_text(
        "[classifiers.svm_rbf]\nC = 3.0\n[model]\npca_components = 8\n"
    )
    config = apply_round(make_config(workspace, tmp_path / "out"), round_file)
    before = npz_files(workspace["root"] / "cache")

    again = train(config, now=NOW)

    assert npz_files(workspace["root"] / "cache") == before
    assert again.report["config"]["classifiers"]["svm_rbf"]["C"] == 3.0
    assert again.report["config"]["model"]["pca_components"] == 8
    assert again.report["config_source"]["path"] == round_file.as_posix()


def test_a_preprocessing_round_recomputes_features(
    workspace: dict[str, Path], result: TrainingResult, tmp_path: Path
) -> None:
    round_file = tmp_path / "round.toml"
    round_file.write_text("[preprocessing]\ncrop_margin = 0.1\n")
    before = npz_files(workspace["root"] / "cache")
    config = apply_round(make_config(workspace, tmp_path / "out"), round_file)

    train(config, now=NOW)

    assert set(npz_files(workspace["root"] / "cache")) > set(before)
