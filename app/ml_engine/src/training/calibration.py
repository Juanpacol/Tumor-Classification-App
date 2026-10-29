"""Probability calibration of the selected classifier."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.training.model import DEFAULT_SEED, N_SPLITS, group_folds

CALIBRATION_METHOD = "sigmoid"


def calibrate(
    model: Pipeline,
    matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = DEFAULT_SEED,
    n_splits: int = N_SPLITS,
) -> CalibratedClassifierCV:
    """Fit the pipeline and its probability calibration on the same group-aware folds.

    Using the selection folds keeps a group of near-duplicate images out of both
    sides of any split, so the calibration cannot reintroduce the leakage. The
    result is an ensemble with one fitted pipeline per fold.
    """
    splits = list(group_folds(n_splits, seed).split(matrix, y, groups))
    calibrated = CalibratedClassifierCV(
        estimator=model, method=CALIBRATION_METHOD, cv=splits
    )
    return calibrated.fit(matrix, y)
