"""Probability calibration of the selected classifier."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import Pipeline

from src.contracts import TRIAGE_CONFIDENCE_THRESHOLD
from src.training.folds import Splits

CALIBRATION_ENSEMBLE = True


def calibrate(
    model: Pipeline,
    matrix: np.ndarray,
    y: np.ndarray,
    splits: Splits,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Fit the pipeline and its probability calibration on the shared folds.

    Each fold fits one pipeline on its fit rows and one calibrator (``method``,
    sigmoid or isotonic) on its validation rows (``CALIBRATION_ENSEMBLE``). The result averages the probabilities of those
    calibrated pipelines, so it is never refitted on the whole Train split.
    Using the selection folds keeps a group of near-duplicate images out of both
    sides of any split.
    """
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=splits,
        ensemble=CALIBRATION_ENSEMBLE,
    )
    return calibrated.fit(matrix, y)


@dataclass(frozen=True)
class CalibratedOutOfFold:
    """Pooled out-of-fold metrics of the calibrated selected classifier on Train."""

    method: str
    macro_f1: float
    log_loss: float
    triage_threshold: float
    confident_share: float
    confident_accuracy: float | None


def calibrated_out_of_fold(
    model: Pipeline,
    matrix: np.ndarray,
    y: np.ndarray,
    splits: Splits,
    n_originals: int,
    method: str,
    labels: Sequence[str],
) -> CalibratedOutOfFold:
    """Calibrated probabilities of every original row by a model that never saw it.

    For outer fold k, the model is calibrated as the final model is, but on the
    other folds only: each of them serves once as calibration rows while the rest
    are fit rows (the shared splits restricted to fold k's fit rows). So no fold-k
    image fits or calibrates the model that predicts it, and Testing is not read.
    """
    labels = sorted(labels)  # log_loss reads probability columns in sorted order
    probabilities = np.zeros((n_originals, len(labels)))
    for fit_rows, validation_rows in splits:
        inner = _inner_splits(fit_rows, validation_rows, splits)
        calibrated = calibrate(model, matrix[fit_rows], y[fit_rows], inner, method)
        order = [list(calibrated.classes_).index(label) for label in labels]
        probabilities[validation_rows] = calibrated.predict_proba(
            matrix[validation_rows]
        )[:, order]
    truth = y[:n_originals]
    predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
    confident = probabilities.max(axis=1) >= TRIAGE_CONFIDENCE_THRESHOLD
    return CalibratedOutOfFold(
        method=method,
        macro_f1=float(f1_score(truth, predicted, labels=labels, average="macro")),
        log_loss=float(log_loss(truth, probabilities, labels=labels)),
        triage_threshold=TRIAGE_CONFIDENCE_THRESHOLD,
        confident_share=float(confident.mean()),
        confident_accuracy=(
            float((predicted[confident] == truth[confident]).mean())
            if confident.any()
            else None
        ),
    )


def _inner_splits(
    fit_rows: np.ndarray, validation_rows: np.ndarray, splits: Splits
) -> Splits:
    """The shared splits of the other folds, as positions inside ``fit_rows``."""
    position = {row: index for index, row in enumerate(fit_rows)}
    inner = []
    for other_fit, other_validation in splits:
        if np.array_equal(other_validation, validation_rows):
            continue
        inner_fit = np.intersect1d(other_fit, fit_rows)
        inner.append(
            (
                np.array([position[row] for row in inner_fit]),
                np.array([position[row] for row in other_validation]),
            )
        )
    return inner
