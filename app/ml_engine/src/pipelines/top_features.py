"""The radiomic features that stand out most against the training data."""

from __future__ import annotations

import numpy as np

from src.contracts import FeatureContribution
from src.training.stats import FeatureStats

_FLAT = 1e-12


def top_features(
    values: np.ndarray, stats: FeatureStats, count: int = 5
) -> tuple[FeatureContribution, ...]:
    """The ``count`` features with the largest |z-score| against the training statistics.

    A feature that never varied in training (std 0) has z-score 0. Ties are broken
    by feature name so the order is stable.
    """
    contributions = [
        FeatureContribution(
            name=name,
            value=float(value),
            z_score=0.0 if std <= _FLAT else float((value - mean) / std),
        )
        for name, value, mean, std in zip(stats.names, values, stats.mean, stats.std)
    ]
    contributions.sort(key=lambda c: (-abs(c.z_score), c.name))
    return tuple(contributions[:count])
