"""Where the out-of-fold errors on Train fall: per image and by subgroup."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.training.dataset import Sample
from src.training.shortcut import SOURCE_SIZE, SOURCE_SIZE_NAME, metadata_matrix

QUALITY_METRICS = ("snr", "entropy")
TERCILES = ("low", "mid", "high")
PREDICTIONS_FILE = "oof_predictions.csv"


def predictions_table(
    samples: Sequence[Sample],
    data_dir: Path,
    folds: np.ndarray,
    quality: np.ndarray,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    """One row per usable original Train image with its out-of-fold prediction.

    ``folds``, ``quality`` and ``predictions`` are aligned with ``samples``.
    Group size counts the usable Train images of the same ``group_id``.
    """
    sizes = metadata_matrix(samples)
    group_sizes = Counter(s.group_id for s in samples)
    return [
        {
            "path": sample.path.relative_to(data_dir).as_posix(),
            "label": sample.label.value,
            "fold": int(fold),
            "group_id": sample.group_id,
            "group_size": group_sizes[sample.group_id],
            "width": int(size[0]),
            "height": int(size[1]),
            "snr": float(metrics[0]),
            "entropy": float(metrics[1]),
            "predicted": str(predicted),
        }
        for sample, fold, metrics, predicted, size in zip(
            samples, folds, quality, predictions, sizes, strict=True
        )
    ]


def error_breakdown(
    rows: Sequence[dict[str, Any]], labels: Sequence[str], no_tumor: str
) -> dict[str, Any]:
    """Recall per class and tumor images predicted ``no_tumor``, per subgroup.

    Each split (source size, group size, snr and entropy terciles) partitions the
    rows, so its cells add up to every image. Tercile edges come from these Train
    rows and are recorded.
    """
    y = np.array([row["label"] for row in rows])
    predicted = np.array([row["predicted"] for row in rows])
    is_source = np.array(
        [(row["width"], row["height"]) == SOURCE_SIZE for row in rows], dtype=bool
    )
    grouped = np.array([row["group_size"] > 1 for row in rows], dtype=bool)
    splits: dict[str, dict[str, np.ndarray]] = {
        "source_size": {SOURCE_SIZE_NAME: is_source, "other": ~is_source},
        "group_size": {"single": ~grouped, "grouped": grouped},
    }
    edges = {}
    for metric in QUALITY_METRICS:
        values = np.array([row[metric] for row in rows], dtype=float)
        low, high = np.quantile(values, [1 / 3, 2 / 3]) if len(values) else (0.0, 0.0)
        edges[metric] = [float(low), float(high)]
        splits[f"{metric}_tercile"] = {
            "low": values < low,
            "mid": (values >= low) & (values < high),
            "high": values >= high,
        }
    return {
        "tercile_edges": edges,
        "splits": {
            name: {
                cell: _cell(y[mask], predicted[mask], labels, no_tumor)
                for cell, mask in cells.items()
            }
            for name, cells in splits.items()
        },
    }


def _cell(
    y: np.ndarray, predicted: np.ndarray, labels: Sequence[str], no_tumor: str
) -> dict[str, Any]:
    recall = {}
    for label in labels:
        rows = y == label
        recall[label] = float(np.mean(predicted[rows] == label)) if rows.any() else None
    return {
        "images": len(y),
        "recall": recall,
        "tumor_as_no_tumor": int(np.sum((y != no_tumor) & (predicted == no_tumor))),
    }
