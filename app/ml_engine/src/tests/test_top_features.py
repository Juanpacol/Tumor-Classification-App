import numpy as np
import pytest

from src.pipelines.top_features import top_features
from src.training.stats import FeatureStats

STATS = FeatureStats(
    names=("a", "b", "c", "d", "e", "f", "g"),
    mean=(0.0,) * 7,
    std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
)


def test_the_largest_absolute_z_scores_come_first() -> None:
    values = np.array([0.5, -3.0, 2.0, 0.1, -1.0, 0.0, 9.0])

    features = top_features(values, STATS)

    assert [f.name for f in features] == ["b", "c", "e", "a", "d"]
    assert [f.z_score for f in features] == pytest.approx([-3.0, 2.0, -1.0, 0.5, 0.1])


def test_value_is_the_raw_feature_and_z_score_uses_mean_and_std() -> None:
    stats = FeatureStats(("x",), (10.0,), (2.0,))

    (feature,) = top_features(np.array([16.0]), stats)

    assert feature.value == 16.0
    assert feature.z_score == pytest.approx(3.0)


def test_a_feature_that_never_varied_has_z_score_zero() -> None:
    features = top_features(np.array([0, 0, 0, 0, 0, 0, 9.0]), STATS, count=7)

    flat = next(f for f in features if f.name == "g")
    assert flat.z_score == 0.0 and flat.value == 9.0


def test_ties_are_broken_by_name() -> None:
    values = np.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    assert [f.name for f in top_features(values, STATS, count=3)] == ["a", "b", "c"]


def test_count_limits_the_result() -> None:
    assert len(top_features(np.ones(7), STATS, count=2)) == 2
    assert len(top_features(np.ones(7), STATS)) == 5
