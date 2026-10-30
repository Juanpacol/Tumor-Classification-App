import numpy as np
import pytest

from src.contracts import TumorLabel
from src.training.evaluation import evaluate

LABELS = [label.value for label in TumorLabel]


class FixedPredictions:
    def __init__(self, predictions: list[str]) -> None:
        self._predictions = np.array(predictions)

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return self._predictions


def test_perfect_predictions() -> None:
    y = np.array(LABELS * 2)

    metrics = evaluate(FixedPredictions(list(y)), np.zeros((8, 1)), y, LABELS)

    assert metrics.accuracy == 1.0 and metrics.macro_f1 == 1.0
    assert metrics.confusion_matrix == [
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 2],
    ]


def test_known_mixed_predictions() -> None:
    y = np.array(["glioma", "glioma", "meningioma", "no_tumor"])
    predictions = ["glioma", "meningioma", "meningioma", "no_tumor"]

    metrics = evaluate(FixedPredictions(predictions), np.zeros((4, 1)), y, LABELS)

    assert metrics.accuracy == pytest.approx(0.75)
    glioma = metrics.per_class["glioma"]
    assert (glioma.precision, glioma.recall, glioma.support) == (1.0, 0.5, 2)
    assert glioma.f1 == pytest.approx(2 / 3)
    assert metrics.per_class["meningioma"].precision == pytest.approx(0.5)
    assert metrics.per_class["pituitary"].support == 0
    assert metrics.per_class["pituitary"].f1 == 0.0
    assert metrics.macro_f1 == pytest.approx((2 / 3 + 2 / 3 + 1.0 + 0.0) / 4)


def test_confusion_matrix_rows_are_the_truth_in_label_order() -> None:
    y = np.array(["glioma", "glioma", "pituitary"])
    predictions = ["glioma", "meningioma", "pituitary"]

    matrix = evaluate(
        FixedPredictions(predictions), np.zeros((3, 1)), y, LABELS
    ).confusion_matrix

    assert matrix[0] == [1, 1, 0, 0]
    index = LABELS.index("pituitary")
    assert matrix[index] == [1 if i == index else 0 for i in range(4)]


def test_label_order_is_respected() -> None:
    y = np.array(["glioma", "pituitary"])

    metrics = evaluate(
        FixedPredictions(list(y)), np.zeros((2, 1)), y, ["pituitary", "glioma"]
    )

    assert list(metrics.per_class) == ["pituitary", "glioma"]


def test_balanced_accuracy_averages_the_recall_of_the_classes_present() -> None:
    y = np.array(["glioma", "glioma", "glioma", "no_tumor"])
    predictions = ["glioma", "glioma", "glioma", "glioma"]

    metrics = evaluate(FixedPredictions(predictions), np.zeros((4, 1)), y, LABELS)

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.balanced_accuracy == pytest.approx(0.5)
