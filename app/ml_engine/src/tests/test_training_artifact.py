import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from src.contracts import TumorLabel
from src.deep_learning.bundle import MANIFEST_FILE as BACKBONE_MANIFEST
from src.deep_learning.bundle import WEIGHTS_FILE as BACKBONE_WEIGHTS
from src.deep_learning.bundle import sha256_of, write_bundle
from src.errors import ModelUnavailableError
from src.tests.synthetic import separable_features
from src.training.artifact import (
    CLASSIFIER_FILE,
    MANIFEST_FILE,
    STATS_FILE,
    write_artifact,
)
from src.training.model import build_model
from src.training.stats import feature_stats

N_RADIOMIC, N_CNN = 45, 20
FIELDS = {"version": "1.2.3", "seed": 42}


@pytest.fixture()
def backbone(tmp_path: Path) -> Path:
    directory = tmp_path / "backbone"
    write_bundle(directory, "efficientnet_b0", {"w": torch.ones(2)}, "TEST")
    return directory


@pytest.fixture()
def fitted() -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    matrix, labels, _ = separable_features(n_per_class=20)
    model = build_model(LogisticRegression(max_iter=500), N_RADIOMIC, N_CNN)
    return model.fit(matrix, labels), matrix, labels


def stats() -> object:
    return feature_stats(np.ones((3, 2)), ("a", "b"))


def test_artifact_folder_has_every_file(
    tmp_path: Path, backbone: Path, fitted: tuple
) -> None:
    model = fitted[0]

    write_artifact(tmp_path / "model", model, backbone, stats(), FIELDS)

    names = {p.name for p in (tmp_path / "model").iterdir()}
    assert names == {
        CLASSIFIER_FILE,
        STATS_FILE,
        MANIFEST_FILE,
        BACKBONE_WEIGHTS,
        BACKBONE_MANIFEST,
    }


def test_manifest_describes_the_artifact(
    tmp_path: Path, backbone: Path, fitted: tuple
) -> None:
    directory = tmp_path / "model"

    manifest = write_artifact(directory, fitted[0], backbone, stats(), FIELDS)

    assert manifest == json.loads((directory / MANIFEST_FILE).read_text())
    assert manifest["version"] == "1.2.3" and manifest["seed"] == 42
    assert manifest["classes"] == [label.value for label in TumorLabel]
    assert manifest["backbone"] == "efficientnet_b0"
    assert {"python", "scikit-learn", "torch", "numpy"} <= set(manifest["libraries"])


def test_manifest_hashes_match_the_files(
    tmp_path: Path, backbone: Path, fitted: tuple
) -> None:
    directory = tmp_path / "model"

    manifest = write_artifact(directory, fitted[0], backbone, stats(), FIELDS)

    assert set(manifest["sha256"]) == {
        CLASSIFIER_FILE,
        STATS_FILE,
        BACKBONE_WEIGHTS,
        BACKBONE_MANIFEST,
    }
    for name, digest in manifest["sha256"].items():
        assert sha256_of(directory / name) == digest


def test_classifier_loads_back_and_predicts_the_same(
    tmp_path: Path, backbone: Path, fitted: tuple
) -> None:
    model, matrix, _ = fitted

    write_artifact(tmp_path / "model", model, backbone, stats(), FIELDS)

    loaded = joblib.load(tmp_path / "model" / CLASSIFIER_FILE)
    assert np.array_equal(loaded.predict(matrix), model.predict(matrix))


def test_feature_statistics_are_stored_for_z_scores(
    tmp_path: Path, backbone: Path, fitted: tuple
) -> None:
    write_artifact(tmp_path / "model", fitted[0], backbone, stats(), FIELDS)

    saved = json.loads((tmp_path / "model" / STATS_FILE).read_text())

    assert saved == {"names": ["a", "b"], "mean": [1.0, 1.0], "std": [0.0, 0.0]}


def test_backbone_files_are_copied_unchanged(
    tmp_path: Path, backbone: Path, fitted: tuple
) -> None:
    write_artifact(tmp_path / "model", fitted[0], backbone, stats(), FIELDS)

    for name in (BACKBONE_WEIGHTS, BACKBONE_MANIFEST):
        assert (tmp_path / "model" / name).read_bytes() == (
            backbone / name
        ).read_bytes()


def test_an_existing_artifact_is_never_overwritten(
    tmp_path: Path, backbone: Path, fitted: tuple
) -> None:
    write_artifact(tmp_path / "model", fitted[0], backbone, stats(), FIELDS)

    with pytest.raises(FileExistsError):
        write_artifact(tmp_path / "model", fitted[0], backbone, stats(), FIELDS)


def test_a_tampered_backbone_is_refused_before_anything_is_written(
    tmp_path: Path, backbone: Path, fitted: tuple
) -> None:
    with (backbone / BACKBONE_WEIGHTS).open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ModelUnavailableError, match="sha256"):
        write_artifact(tmp_path / "model", fitted[0], backbone, stats(), FIELDS)

    assert not (tmp_path / "model").exists()


def test_manifest_carries_the_backbone_provenance(
    tmp_path: Path, fitted: tuple
) -> None:
    source = tmp_path / "source"
    write_bundle(
        source,
        "efficientnet_b0",
        {"w": torch.ones(2)},
        "TEST_V1",
        source_url="https://example.org/weights.pth",
        license_note="a note",
    )

    manifest = write_artifact(tmp_path / "model", fitted[0], source, stats(), FIELDS)

    assert manifest["backbone_provenance"] == {
        "weights_version": "TEST_V1",
        "source_url": "https://example.org/weights.pth",
        "license_note": "a note",
    }
