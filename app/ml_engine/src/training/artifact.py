"""Versioned model artifact: classifier, backbone, feature statistics and manifest."""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
from sklearn.base import BaseEstimator

from src.contracts import TumorLabel
from src.deep_learning.bundle import MANIFEST_FILE as BACKBONE_MANIFEST
from src.deep_learning.bundle import WEIGHTS_FILE as BACKBONE_WEIGHTS
from src.deep_learning.bundle import read_bundle, sha256_of
from src.training.stats import FeatureStats

CLASSIFIER_FILE = "classifier.joblib"
STATS_FILE = "feature_stats.json"
MANIFEST_FILE = "manifest.json"
_LIBRARIES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "scikit-image",
    "PyWavelets",
    "joblib",
    "torch",
    "torchvision",
)


def write_artifact(
    directory: Path,
    model: BaseEstimator,
    backbone_bundle: Path,
    stats: FeatureStats,
    manifest_fields: dict[str, Any],
) -> dict[str, Any]:
    """Write the artifact folder and return its manifest.

    The backbone bundle is verified before it is copied. The manifest holds
    ``manifest_fields`` plus the class names, the library versions and the
    sha256 of each file, so a consumer can check integrity before loading the
    pickled classifier. Raises ``FileExistsError`` rather than overwrite.
    """
    backbone = read_bundle(backbone_bundle)[1]
    directory.mkdir(parents=True)

    joblib.dump(model, directory / CLASSIFIER_FILE)
    (directory / STATS_FILE).write_text(json.dumps(asdict(stats), indent=2))
    for name in (BACKBONE_WEIGHTS, BACKBONE_MANIFEST):
        shutil.copy2(backbone_bundle / name, directory / name)

    manifest = {
        **manifest_fields,
        "classes": [label.value for label in TumorLabel],
        "backbone": backbone.backbone,
        "backbone_provenance": {
            "weights_version": backbone.weights_version,
            "source_url": backbone.source_url,
            "license_note": backbone.license_note,
        },
        "libraries": {
            "python": platform.python_version(),
            **{name: version(name) for name in _LIBRARIES},
        },
        "sha256": {
            name: sha256_of(directory / name)
            for name in (
                CLASSIFIER_FILE,
                STATS_FILE,
                BACKBONE_WEIGHTS,
                BACKBONE_MANIFEST,
            )
        },
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return manifest
