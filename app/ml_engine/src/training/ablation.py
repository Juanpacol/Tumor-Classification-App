"""Which feature blocks the classifier needs."""

from __future__ import annotations

import numpy as np

from src.training.folds import Splits
from src.training.model import (
    DEFAULT_SEED,
    candidate_classifiers,
    cross_validate_scores,
)

SINGLE_BLOCK_VARIANTS = {"cnn_only": ("cnn",), "radiomics_only": ("radiomic",)}
VARIANTS = (*SINGLE_BLOCK_VARIANTS, "hybrid")


def ablate_blocks(
    matrix: np.ndarray,
    y: np.ndarray,
    n_radiomic: int,
    classifier_name: str,
    splits: Splits,
    hybrid_macro_f1: float,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Mean cross-validated macro-F1 of one classifier per feature variant.

    ``hybrid_macro_f1`` is the selection score of the same classifier on the same
    splits, which is the hybrid variant, so it is reused instead of refitted. It
    uses the training folds only, so the held-out test set is not touched.
    """
    classifier = candidate_classifiers(seed)[classifier_name]
    single = {
        variant: cross_validate_scores(
            classifier, matrix, y, n_radiomic, splits, seed, blocks
        ).mean_macro_f1
        for variant, blocks in SINGLE_BLOCK_VARIANTS.items()
    }
    return {**single, "hybrid": hybrid_macro_f1}
