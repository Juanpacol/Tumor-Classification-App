"""Out-of-fold predictions and per-class metrics of the selected pipeline on Train."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator

from src.training.evaluation import EvaluationMetrics, evaluate


def out_of_fold_predictions(
    estimators: Sequence[BaseEstimator],
    validation_rows: Sequence[np.ndarray],
    matrix: np.ndarray,
    y: np.ndarray,
    n_originals: int,
) -> np.ndarray:
    """Predict every original row with the estimator fitted without its fold.

    ``estimators[k]`` was fitted on fold k's fit rows and predicts its
    ``validation_rows[k]``, so nothing is refitted. Originals are the first
    ``n_originals`` rows; the rows after them are augmented copies, which may be fit
    rows but are never predicted. Raises ``ValueError`` if a row is validated twice,
    a copy is validated, or an original is left out.
    """
    predictions = np.empty_like(y[:n_originals])
    validated = np.zeros(len(y), dtype=bool)
    for estimator, rows in zip(estimators, validation_rows, strict=True):
        if validated[rows].any():
            raise ValueError("a row is validated by more than one fold")
        validated[rows] = True
        if rows.size and rows.max() >= n_originals:
            raise ValueError("an augmented copy is a validation row")
        predictions[rows] = estimator.predict(matrix[rows])
    if not validated[:n_originals].all():
        raise ValueError("an original row is in no validation fold")
    return predictions


def out_of_fold_metrics(
    y: np.ndarray, predictions: np.ndarray, labels: Sequence[str]
) -> EvaluationMetrics:
    """Per-class scores and confusion matrix of the out-of-fold predictions of the
    originals ``y``, pooled over folds. Uncalibrated: the calibrated model fits its
    sigmoids on these rows."""
    return evaluate(y, predictions, labels)
