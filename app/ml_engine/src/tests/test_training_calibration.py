import numpy as np
from pytest import MonkeyPatch
from sklearn.linear_model import LogisticRegression

from src.tests.synthetic import grouped_splits, separable_features
from src.training import calibration
from src.training.calibration import calibrate
from src.training.model import build_model

N_RADIOMIC, N_CNN = 45, 20


def fresh_model() -> object:
    return build_model(LogisticRegression(max_iter=500), N_RADIOMIC, N_CNN)


def test_calibrated_probabilities_are_valid_and_follow_the_class_order() -> None:
    matrix, labels, groups = separable_features()

    calibrated = calibrate(
        fresh_model(), matrix, labels, grouped_splits(labels, groups, 3)
    )

    probabilities = calibrated.predict_proba(matrix)
    assert probabilities.shape == (len(labels), 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert list(calibrated.classes_) == sorted(set(labels))
    assert (calibrated.predict(matrix) == labels).mean() > 0.9


def test_calibration_is_reproducible() -> None:
    matrix, labels, groups = separable_features()
    splits = grouped_splits(labels, groups, 3, seed=3)

    first = calibrate(fresh_model(), matrix, labels, splits)
    second = calibrate(fresh_model(), matrix, labels, splits)

    assert np.allclose(first.predict_proba(matrix), second.predict_proba(matrix))


def test_the_result_is_one_calibrated_pipeline_per_fold() -> None:
    matrix, labels, groups = separable_features()

    calibrated = calibrate(
        fresh_model(), matrix, labels, grouped_splits(labels, groups, 3)
    )

    assert len(calibrated.calibrated_classifiers_) == 3
    expected = np.mean(
        [c.predict_proba(matrix) for c in calibrated.calibrated_classifiers_], axis=0
    )
    assert np.allclose(calibrated.predict_proba(matrix), expected)


def test_calibration_uses_the_given_splits_as_they_are(
    monkeypatch: MonkeyPatch,
) -> None:
    matrix, labels, groups = separable_features()
    splits = grouped_splits(labels, groups, 3)
    seen: list[object] = []
    real = calibration.CalibratedClassifierCV

    def recording(**kwargs: object) -> object:
        seen.append(kwargs["cv"])
        return real(**kwargs)

    monkeypatch.setattr(calibration, "CalibratedClassifierCV", recording)

    calibrate(fresh_model(), matrix, labels, splits)

    assert seen == [splits]
