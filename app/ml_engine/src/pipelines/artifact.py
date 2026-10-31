"""Read and verify a training artifact for serving."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.base import BaseEstimator

from src.classical.config import RadiomicsConfig
from src.classical.extractor import feature_names
from src.contracts import TumorLabel
from src.deep_learning.bundle import sha256_of
from src.errors import ModelUnavailableError
from src.preprocessing.prepare import PreprocessingConfig
from src.preprocessing.quality import METRICS_VERSION, QualityThresholds
from src.training.artifact import CLASSIFIER_FILE, MANIFEST_FILE, STATS_FILE
from src.training.stats import FeatureStats


@dataclass(frozen=True)
class LoadedArtifact:
    """What serving needs, rebuilt from an artifact folder."""

    classifier: BaseEstimator
    stats: FeatureStats
    preprocessing: PreprocessingConfig
    radiomics: RadiomicsConfig
    version: str


def load_artifact(directory: Path) -> LoadedArtifact:
    """Verify every file against the manifest, then load the classifier.

    The sha256 of each file is checked before the pickled classifier is
    deserialized. The preprocessing and radiomics settings come from the
    manifest, so serving uses the ones the model was trained with. Raises
    ``ModelUnavailableError`` when anything is missing, altered or inconsistent.
    """
    manifest = _read_manifest(directory)
    _verify_hashes(directory, manifest)
    try:
        preprocessing = _preprocessing(manifest["preprocessing"])
        radiomics = RadiomicsConfig(**_tuples(manifest["radiomics"]))
        stats = FeatureStats(
            **_tuples(json.loads((directory / STATS_FILE).read_text()))
        )
        version = str(manifest["version"])
        classes = manifest["classes"]
        names = manifest["radiomic_feature_names"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelUnavailableError(f"invalid artifact manifest: {exc}") from exc

    if classes != [label.value for label in TumorLabel]:
        raise ModelUnavailableError("artifact classes do not match this engine")
    if list(stats.names) != names or tuple(names) != feature_names(radiomics):
        raise ModelUnavailableError("artifact feature names do not match this engine")

    try:
        classifier = joblib.load(directory / CLASSIFIER_FILE)
    except Exception as exc:
        raise ModelUnavailableError(f"cannot load classifier: {exc}") from exc
    if set(classifier.classes_) != set(classes):
        raise ModelUnavailableError("classifier classes do not match the manifest")
    return LoadedArtifact(classifier, stats, preprocessing, radiomics, version)


def _read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise ModelUnavailableError(f"artifact manifest not found: {path}")
    try:
        manifest: dict[str, Any] = json.loads(path.read_text())
    except ValueError as exc:
        raise ModelUnavailableError(f"invalid artifact manifest: {exc}") from exc
    return manifest


def _verify_hashes(directory: Path, manifest: dict[str, Any]) -> None:
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or CLASSIFIER_FILE not in hashes:
        raise ModelUnavailableError("artifact manifest has no file hashes")
    for name, expected in hashes.items():
        path = directory / Path(name).name
        if not path.is_file():
            raise ModelUnavailableError(f"artifact file not found: {path}")
        if sha256_of(path) != expected:
            raise ModelUnavailableError(f"{name} does not match the manifest sha256")


def _tuples(values: dict[str, Any]) -> dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}


def _preprocessing(values: dict[str, Any]) -> PreprocessingConfig:
    fields = _tuples(values)
    version = fields["thresholds"].get("metrics_version")
    if version != METRICS_VERSION:
        raise ModelUnavailableError(
            f"artifact quality metrics version {version} does not match "
            f"this engine ({METRICS_VERSION}); retrain the model"
        )
    fields["thresholds"] = QualityThresholds(**fields["thresholds"])
    return PreprocessingConfig(**fields)
