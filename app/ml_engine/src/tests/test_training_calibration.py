import numpy as np
from pytest import MonkeyPatch
from sklearn.linear_model import LogisticRegression

from src.tests.synthetic import separable_features
from src.training import calibration
from src.training.calibration import calibrate
from src.training.model import build_model, group_folds

N_RADIOMIC, N_CNN = 45, 20


def fresh_model() -> object:
    return build_model(LogisticRegression(max_iter=500), N_RADIOMIC, N_CNN)


def test_calibrated_probabilities_are_valid_and_follow_the_class_order() -> None:
    matrix, labels, groups = separable_features()

    calibrated = calibrate(fresh_model(), matrix, labels, groups, n_splits=3)

    probabilities = calibrated.predict_proba(matrix)
    assert probabilities.shape == (len(labels), 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert list(calibrated.classes_) == sorted(set(labels))
    assert (calibrated.predict(matrix) == labels).mean() > 0.9


def test_calibration_is_reproducible() -> None:
    matrix, labels, groups = separable_features()

    first = calibrate(fresh_model(), matrix, labels, groups, seed=3, n_splits=3)
    second = calibrate(fresh_model(), matrix, labels, groups, seed=3, n_splits=3)

    assert np.allclose(first.predict_proba(matrix), second.predict_proba(matrix))


def test_calibration_never_splits_a_group_across_folds(
    monkeypatch: MonkeyPatch,
) -> None:
    matrix, labels, groups = separable_features()
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    class RecordingFolds:
        def split(self, x: np.ndarray, y: np.ndarray, g: np.ndarray) -> object:
            splits = list(group_folds(3, 0).split(x, y, g))
            seen.extend(splits)
            return iter(splits)

    monkeypatch.setattr(
        calibration, "group_folds", lambda n_splits, seed: RecordingFolds()
    )

    calibrate(fresh_model(), matrix, labels, groups, n_splits=3)

    assert len(seen) == 3
    for train, test in seen:
        assert not set(groups[train]) & set(groups[test])
