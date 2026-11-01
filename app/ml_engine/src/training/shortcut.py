"""Shortcut audit (block B10): how far the class can be told from the file alone."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.training.dataset import Sample
from src.training.folds import Splits

METADATA_FEATURES = ("width", "height", "file_size")
WARNING_BALANCED_ACCURACY = 0.70
SOURCE_SIZE = (512, 512)
SOURCE_SIZE_NAME = "512x512"


def metadata_matrix(samples: Sequence[Sample]) -> np.ndarray:
    """Width, height and file size of every sample, read from the file header."""
    rows = []
    for sample in samples:
        with Image.open(sample.path) as image:
            width, height = image.size
        rows.append((width, height, sample.path.stat().st_size))
    return np.array(rows, dtype=float).reshape(len(rows), len(METADATA_FEATURES))


def metadata_baseline(
    samples: Sequence[Sample], y: np.ndarray, splits: Splits, seed: int
) -> dict[str, Any]:
    """Cross-validated balanced accuracy of a classifier that sees only file metadata.

    It uses the same rows and splits as the hybrid model, so a high score means the
    folds can be separated without any anatomy. A random forest is used because
    the cue is a threshold on size (for example exactly 512x512), which a linear
    model would miss.
    """
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    scores = cross_val_score(
        model, metadata_matrix(samples), y, cv=splits, scoring="balanced_accuracy"
    )
    mean = float(scores.mean())
    return {
        "features": list(METADATA_FEATURES),
        "classifier": "random_forest",
        "mean_balanced_accuracy": mean,
        "std_balanced_accuracy": float(scores.std()),
        "warning_level": WARNING_BALANCED_ACCURACY,
        "warning": mean >= WARNING_BALANCED_ACCURACY,
    }


def recall_by_source_size(
    samples: Sequence[Sample],
    y: np.ndarray,
    predictions: np.ndarray,
    label: str,
) -> dict[str, dict[str, Any]]:
    """Recall of ``label`` for its 512x512 images and for the others.

    ``recall`` is ``None`` for a group without images of ``label``.
    """
    sizes = metadata_matrix(samples)[:, :2]
    is_source_size = np.all(sizes == SOURCE_SIZE, axis=1)
    of_label = y == label
    groups = {SOURCE_SIZE_NAME: is_source_size, "other": ~is_source_size}
    result = {}
    for name, in_group in groups.items():
        rows = of_label & in_group
        support = int(rows.sum())
        recall = float(np.mean(predictions[rows] == label)) if support else None
        result[name] = {"recall": recall, "support": support}
    return result
