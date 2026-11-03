import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from src.tests.synthetic import separable_features
from src.training.folds import fold_splits
from src.training.model import build_model, cross_validate_predictions
from src.training.out_of_fold import out_of_fold_metrics, out_of_fold_predictions

N_RADIOMIC, N_CNN, N_SPLITS = 45, 20, 3


class FoldTag:
    """A fitted stand-in that predicts its own fold number for every row."""

    def __init__(self, fold: int) -> None:
        self.fold = fold

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.fold)


def data_with_copies() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Originals followed by one copy each, every copy in its original's fold."""
    matrix, labels, _ = separable_features(n_per_class=12)
    n = len(labels)
    fold_of_row = np.arange(n) % N_SPLITS
    return (
        np.vstack([matrix, matrix]),
        np.concatenate([labels, labels]),
        np.concatenate([fold_of_row, fold_of_row]),
        n,
    )


def copy_splits(folds: np.ndarray, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return fold_splits(folds, N_SPLITS, np.arange(len(folds)) >= n)


def test_every_original_is_predicted_once_by_its_own_fold_model() -> None:
    _, _, folds, n = data_with_copies()
    splits = copy_splits(folds, n)
    validation = [rows for _, rows in splits]

    predictions = out_of_fold_predictions(
        [FoldTag(k) for k in range(N_SPLITS)],
        validation,
        np.zeros((len(folds), 1)),
        np.zeros(len(folds), dtype=int),
        n,
    )

    assert predictions.shape == (n,)
    assert np.array_equal(predictions, folds[:n])
    assert sorted(np.concatenate(validation)) == list(range(n))


def test_predictions_match_a_model_fitted_without_the_fold() -> None:
    matrix, labels, folds, n = data_with_copies()
    splits = copy_splits(folds, n)
    model = build_model(LogisticRegression(max_iter=500), N_RADIOMIC, N_CNN)

    _, predictions = cross_validate_predictions(
        LogisticRegression(max_iter=500), matrix, labels, N_RADIOMIC, splits, n
    )

    for fit, validation in splits:
        fitted = clone(model).fit(matrix[fit], labels[fit])
        assert np.array_equal(
            predictions[validation], fitted.predict(matrix[validation])
        )


def test_metrics_cover_the_usable_originals() -> None:
    matrix, labels, folds, n = data_with_copies()
    classes = sorted(set(labels))

    _, predictions = cross_validate_predictions(
        LogisticRegression(max_iter=500),
        matrix,
        labels,
        N_RADIOMIC,
        copy_splits(folds, n),
        n,
    )
    metrics = out_of_fold_metrics(labels[:n], predictions, classes)

    assert sum(map(sum, metrics.confusion_matrix)) == n
    assert {c: metrics.per_class[c].support for c in classes} == {
        c: int(np.sum(labels[:n] == c)) for c in classes
    }
    assert metrics.macro_f1 > 0.9


def test_splits_that_validate_a_copy_are_rejected() -> None:
    _, _, folds, n = data_with_copies()
    validation = [rows for _, rows in fold_splits(folds, N_SPLITS)]
    tags = [FoldTag(k) for k in range(N_SPLITS)]

    with pytest.raises(ValueError, match="copy"):
        out_of_fold_predictions(tags, validation, np.zeros((2 * n, 1)), folds, n)


def test_splits_that_skip_an_original_are_rejected() -> None:
    _, _, folds, n = data_with_copies()
    validation = [rows for _, rows in copy_splits(folds, n)][:-1]
    tags = [FoldTag(k) for k in range(N_SPLITS - 1)]

    with pytest.raises(ValueError, match="original"):
        out_of_fold_predictions(tags, validation, np.zeros((2 * n, 1)), folds, n)


def test_a_row_validated_twice_is_rejected() -> None:
    _, _, folds, n = data_with_copies()
    validation = [rows for _, rows in copy_splits(folds, n)]
    validation = [*validation, validation[0]]
    tags = [FoldTag(k) for k in range(N_SPLITS + 1)]

    with pytest.raises(ValueError, match="more than one fold"):
        out_of_fold_predictions(tags, validation, np.zeros((2 * n, 1)), folds, n)
