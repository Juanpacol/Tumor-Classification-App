"""Which feature blocks the classifier needs."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from src.training.folds import Splits
from src.training.model import (
    BLOCKS,
    DEFAULT_SEED,
    PCA_COMPONENTS,
    cross_validate_scores,
)

VARIANT_BLOCKS = {
    "cnn_only": ("cnn",),
    "radiomics_only": ("radiomic",),
    "hybrid": BLOCKS,
}
VARIANTS = tuple(VARIANT_BLOCKS)


def ablate_blocks(
    matrix: np.ndarray,
    y: np.ndarray,
    n_radiomic: int,
    classifier: BaseEstimator,
    splits: Splits,
    reference_macro_f1: float,
    seed: int = DEFAULT_SEED,
    pca_components: int = PCA_COMPONENTS,
    reference_blocks: tuple[str, ...] = BLOCKS,
) -> dict[str, float]:
    """Mean cross-validated macro-F1 of one classifier per feature variant.

    ``reference_macro_f1`` is the selection score of the same classifier on the
    same splits with ``reference_blocks`` (the blocks the round uses), so that
    variant is reused instead of refitted. It uses the training folds only, so the
    held-out test set is not touched.
    """
    return {
        variant: (
            reference_macro_f1
            if blocks == reference_blocks
            else cross_validate_scores(
                classifier, matrix, y, n_radiomic, splits, seed, blocks, pca_components
            ).mean_macro_f1
        )
        for variant, blocks in VARIANT_BLOCKS.items()
    }
