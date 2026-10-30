import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.tests.synthetic import grouped_splits, separable_features
from src.training import model as model_module
from src.training.model import (
    CandidateScore,
    build_model,
    candidate_classifiers,
    compare_classifiers,
)

N_RADIOMIC, N_CNN = 45, 20


def test_model_reduces_the_radiomic_block_to_32_components() -> None:
    matrix, labels, _ = separable_features()
    model = build_model(LogisticRegression(max_iter=500), N_RADIOMIC, N_CNN)

    model.fit(matrix, labels)

    features = model.named_steps["features"]
    assert features.transform(matrix).shape == (len(labels), 32 + N_CNN)
    assert model.score(matrix, labels) > 0.9


def test_scalers_are_fitted_on_the_training_data_only() -> None:
    matrix, labels, _ = separable_features()
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


def test_selection_scores_every_candidate_and_picks_the_best() -> None:
    matrix, labels, groups = separable_features()

    result = compare_classifiers(
        matrix, labels, N_RADIOMIC, grouped_splits(labels, groups)
    )

    assert set(result.scores) == set(candidate_classifiers())
    for score in result.scores.values():
        assert len(score.fold_macro_f1) == len(score.fold_balanced_accuracy) == 5
        assert 0.0 <= score.mean_macro_f1 <= 1.0
        assert 0.0 <= score.mean_balanced_accuracy <= 1.0
    best_score = result.scores[result.best].mean_macro_f1
    assert best_score == max(s.mean_macro_f1 for s in result.scores.values())
    assert best_score > 0.9


def test_selection_uses_macro_f1_not_balanced_accuracy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_scores(classifier: object, *args: object, **kwargs: object) -> object:
        name = type(classifier).__name__
        f1, balanced = {
            "LogisticRegression": (0.80, 0.95),
            "SVC": (0.90, 0.70),
            "RandomForestClassifier": (0.85, 0.80),
        }[name]
        return CandidateScore(f1, 0.0, (f1,), balanced, 0.0, (balanced,))

    monkeypatch.setattr(model_module, "cross_validate_scores", fake_scores)

    result = compare_classifiers(np.zeros((4, 2)), np.zeros(4), 1, [])

    assert result.best == "svm_rbf"


def test_selection_is_reproducible_with_the_same_seed() -> None:
    matrix, labels, groups = separable_features()
    splits = grouped_splits(labels, groups, seed=7)

    first = compare_classifiers(matrix, labels, N_RADIOMIC, splits, seed=7)
    second = compare_classifiers(matrix, labels, N_RADIOMIC, splits, seed=7)

    assert first == second


def test_unbalanced_classes_still_score_per_class() -> None:
    matrix, labels, groups = separable_features(n_per_class=40)
    keep = np.concatenate(
        [np.flatnonzero(labels == labels[0])[:8], np.flatnonzero(labels != labels[0])]
    )

    result = compare_classifiers(
        matrix[keep],
        labels[keep],
        N_RADIOMIC,
        grouped_splits(labels[keep], groups[keep]),
    )

    assert result.scores[result.best].mean_macro_f1 > 0.8
