import numpy as np
from pytest import MonkeyPatch
from sklearn.linear_model import LogisticRegression

from src.tests.synthetic import grouped_splits, separable_features
from src.training import calibration
from src.training.calibration import calibrate, calibrated_out_of_fold
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


def test_the_configured_method_reaches_every_calibrator() -> None:
    matrix, labels, groups = separable_features()

    calibrated = calibrate(
        fresh_model(), matrix, labels, grouped_splits(labels, groups, 3), "isotonic"
    )

    assert {c.method for c in calibrated.calibrated_classifiers_} == {"isotonic"}


def test_calibrated_out_of_fold_metrics_are_valid() -> None:
    matrix, labels, groups = separable_features()
    splits = grouped_splits(labels, groups, 3)

    metrics = calibrated_out_of_fold(
        fresh_model(), matrix, labels, splits, len(labels), "sigmoid",
        sorted(set(labels)),
    )  # fmt: skip

    assert metrics.method == "sigmoid" and metrics.macro_f1 > 0.8
    assert metrics.log_loss > 0 and 0 <= metrics.confident_share <= 1
    assert metrics.triage_threshold == 0.75


def test_fold_k_never_fits_or_calibrates_the_model_that_predicts_it(
    monkeypatch: MonkeyPatch,
) -> None:
    matrix, labels, groups = separable_features()
    splits = grouped_splits(labels, groups, 3)
    seen: list[tuple[set[int], set[int]]] = []
    real = calibration.calibrate

    def recording(model, x, y, inner, method):  # type: ignore[no-untyped-def]
        rows = {tuple(row) for row in x}
        seen.append((rows, set()))
        return real(model, x, y, inner, method)

    monkeypatch.setattr(calibration, "calibrate", recording)

    calibrated_out_of_fold(
        fresh_model(), matrix, labels, splits, len(labels), "sigmoid",
        sorted(set(labels)),
    )  # fmt: skip

    assert len(seen) == len(splits)
    for (rows, _), (_, validation) in zip(seen, splits):
        assert not rows & {tuple(row) for row in matrix[validation]}


def test_inner_splits_use_each_other_fold_once_as_calibration_rows() -> None:
    _, labels, groups = separable_features()
    splits = grouped_splits(labels, groups, 3)
    fit_rows, validation_rows = splits[0]

    inner = calibration._inner_splits(fit_rows, validation_rows, splits)

    assert len(inner) == 2
    calibration_rows = sorted(np.concatenate([fit_rows[c] for _, c in inner]))
    assert calibration_rows == sorted(fit_rows)
    for inner_fit, inner_calibration in inner:
        assert not set(fit_rows[inner_fit]) & set(fit_rows[inner_calibration])
