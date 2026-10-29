"""Which feature blocks the classifier needs."""

from __future__ import annotations

import numpy as np

from src.training.model import (
    DEFAULT_SEED,
    N_SPLITS,
    candidate_classifiers,
    cross_validate_macro_f1,
)

VARIANTS = {
    "cnn_only": ("cnn",),
    "radiomics_only": ("radiomic",),
    "hybrid": ("radiomic", "cnn"),
}


def ablate_blocks(
    matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_radiomic: int,
    classifier_name: str,
    seed: int = DEFAULT_SEED,
    n_splits: int = N_SPLITS,
) -> dict[str, float]:
    """Mean cross-validated macro-F1 of one classifier per feature variant.

    It uses the training folds only, so the held-out test set is not touched.
    """
    classifier = candidate_classifiers(seed)[classifier_name]
    return {
        variant: cross_validate_macro_f1(
            classifier, matrix, y, groups, n_radiomic, seed, n_splits, blocks
        ).mean_macro_f1
        for variant, blocks in VARIANTS.items()
    }
