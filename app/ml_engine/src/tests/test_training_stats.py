import numpy as np
import pytest

from src.training.stats import feature_stats


def test_stats_are_the_column_mean_and_population_std() -> None:
    radiomics = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])

    stats = feature_stats(radiomics, ("a", "b"))

    assert stats.names == ("a", "b")
    assert stats.mean == pytest.approx((3.0, 5.0))
    assert stats.std == pytest.approx((np.std([1, 3, 5]), 0.0))
