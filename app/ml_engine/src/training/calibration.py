"""Probability calibration of the selected classifier."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.training.folds import Splits

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_ENSEMBLE = True


def calibrate(
    model: Pipeline, matrix: np.ndarray, y: np.ndarray, splits: Splits
) -> CalibratedClassifierCV:
    """Fit the pipeline and its probability calibration on the shared folds.

    Each fold fits one pipeline on its fit rows and one sigmoid on its validation
    rows (``CALIBRATION_ENSEMBLE``). The result averages the probabilities of those
    calibrated pipelines, so it is never refitted on the whole Train split.
    Using the selection folds keeps a group of near-duplicate images out of both
    sides of any split.
    """
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=CALIBRATION_METHOD,
        cv=splits,
        ensemble=CALIBRATION_ENSEMBLE,
    )
    return calibrated.fit(matrix, y)
