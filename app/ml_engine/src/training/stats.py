"""Training statistics of the radiomic features, used later to compute z-scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureStats:
    """Per-feature mean and population standard deviation (``std`` can be 0)."""

    names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]


def feature_stats(radiomics: np.ndarray, names: tuple[str, ...]) -> FeatureStats:
    """Mean and std of every radiomic column of the training set."""
    return FeatureStats(
        names,
        tuple(float(v) for v in radiomics.mean(axis=0)),
        tuple(float(v) for v in radiomics.std(axis=0)),
    )
