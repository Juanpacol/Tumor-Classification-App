"""Held-out evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationMetrics:
    """Accuracy, macro-F1, per-class scores and the confusion matrix (rows are the truth)."""

    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion_matrix: list[list[int]]


def evaluate(
    model: BaseEstimator, matrix: np.ndarray, y: np.ndarray, labels: Sequence[str]
) -> EvaluationMetrics:
    """Score ``model`` on ``matrix``; ``labels`` fixes the order of the classes."""
    predictions = model.predict(matrix)
    report = classification_report(
        y, predictions, labels=list(labels), output_dict=True, zero_division=0
    )
    return EvaluationMetrics(
        accuracy=float(report["accuracy"]),
        macro_f1=float(report["macro avg"]["f1-score"]),
        per_class={
            label: ClassMetrics(
                float(report[label]["precision"]),
                float(report[label]["recall"]),
                float(report[label]["f1-score"]),
                int(report[label]["support"]),
            )
            for label in labels
        },
        confusion_matrix=confusion_matrix(y, predictions, labels=list(labels)).tolist(),
    )
