import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.training.model import (
    build_model,
    candidate_classifiers,
    compare_classifiers,
    group_folds,
)

N_RADIOMIC, N_CNN = 45, 20


def synthetic_features(
    n_per_class: int = 30, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four separable classes; every image belongs to a group of two near-duplicates."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(4), n_per_class)
    centers = rng.normal(scale=3.0, size=(4, N_RADIOMIC + N_CNN))
    matrix = centers[labels] + rng.normal(size=(len(labels), N_RADIOMIC + N_CNN))
    groups = np.arange(len(labels)) // 2
    return matrix, labels, groups


def test_model_reduces_the_radiomic_block_to_32_components() -> None:
    matrix, labels, _ = synthetic_features()
    model = build_model(LogisticRegression(max_iter=500), N_RADIOMIC, N_CNN)

    model.fit(matrix, labels)

    features = model.named_steps["features"]
    assert features.transform(matrix).shape == (len(labels), 32 + N_CNN)
    assert model.score(matrix, labels) > 0.9


def test_scalers_are_fitted_on_the_training_data_only() -> None:
    matrix, labels, _ = synthetic_features()
    train = slice(0, 80)
    model = build_model(LogisticRegression(max_iter=500), N_RADIOMIC, N_CNN)

    model.fit(matrix[train], labels[train])

    scaler = model.named_steps["features"].named_transformers_["cnn"]
    assert np.allclose(scaler.mean_, matrix[train, N_RADIOMIC:].mean(axis=0))
    assert not np.allclose(scaler.mean_, matrix[:, N_RADIOMIC:].mean(axis=0))


def test_pca_components_are_capped_by_the_radiomic_width() -> None:
    model = build_model(LogisticRegression(max_iter=500), n_radiomic=10, n_cnn=N_CNN)
    matrix = np.random.default_rng(0).normal(size=(60, 10 + N_CNN))

    model.fit(matrix, np.arange(60) % 3)

    assert model.named_steps["features"].transform(matrix).shape[1] == 10 + N_CNN


def test_three_classifiers_are_compared_with_balanced_weights() -> None:
    candidates = candidate_classifiers()

    assert set(candidates) == {"logistic_regression", "svm_rbf", "random_forest"}
    assert all(
        c.get_params()["class_weight"] == "balanced" for c in candidates.values()
    )


def test_a_group_never_straddles_two_folds() -> None:
    _, labels, groups = synthetic_features()

    for train, test in group_folds(5, 0).split(np.zeros(len(labels)), labels, groups):
        assert not set(groups[train]) & set(groups[test])


def test_selection_scores_every_candidate_and_picks_the_best() -> None:
    matrix, labels, groups = synthetic_features()

    result = compare_classifiers(matrix, labels, groups, N_RADIOMIC)

    assert set(result.scores) == set(candidate_classifiers())
    assert all(len(s.fold_macro_f1) == 5 for s in result.scores.values())
    assert all(0.0 <= s.mean_macro_f1 <= 1.0 for s in result.scores.values())
    best_score = result.scores[result.best].mean_macro_f1
    assert best_score == max(s.mean_macro_f1 for s in result.scores.values())
    assert best_score > 0.9


def test_selection_is_reproducible_with_the_same_seed() -> None:
    matrix, labels, groups = synthetic_features()

    first = compare_classifiers(matrix, labels, groups, N_RADIOMIC, seed=7)
    second = compare_classifiers(matrix, labels, groups, N_RADIOMIC, seed=7)

    assert first == second


def test_a_different_seed_changes_the_folds() -> None:
    _, labels, groups = synthetic_features()

    def test_sets(seed: int) -> list[list[int]]:
        splits = group_folds(5, seed).split(np.zeros(len(labels)), labels, groups)
        return [sorted(test.tolist()) for _, test in splits]

    assert test_sets(1) != test_sets(2)


def test_unbalanced_classes_still_score_per_class() -> None:
    matrix, labels, groups = synthetic_features(n_per_class=40)
    keep = np.concatenate(
        [np.flatnonzero(labels == 0)[:8], np.flatnonzero(labels != 0)]
    )

    result = compare_classifiers(matrix[keep], labels[keep], groups[keep], N_RADIOMIC)

    assert result.scores[result.best].mean_macro_f1 > 0.8


def test_invalid_split_count_is_rejected() -> None:
    matrix, labels, groups = synthetic_features(n_per_class=4)

    with pytest.raises(ValueError):
        compare_classifiers(matrix, labels, groups, N_RADIOMIC, n_splits=1)
