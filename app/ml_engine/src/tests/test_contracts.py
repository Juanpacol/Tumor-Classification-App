from typing import Any

import pytest

from src.contracts import (
    FeatureContribution,
    PredictionResult,
    QualityMetrics,
    TumorLabel,
)
from src.errors import MLEngineError, QualityCheckError, UnsupportedImageError


def make_result(**overrides: Any) -> PredictionResult:
    fields: dict[str, Any] = {
        "label": TumorLabel.GLIOMA,
        "probabilities": {
            TumorLabel.GLIOMA: 0.7,
            TumorLabel.MENINGIOMA: 0.2,
            TumorLabel.PITUITARY: 0.06,
            TumorLabel.NO_TUMOR: 0.04,
        },
        "confidence": 0.7,
        "needs_expert_triage": False,
        "quality": QualityMetrics(snr=20.0, entropy=6.0, skull_stripped=True),
        "top_features": (
            FeatureContribution(name="glcm_contrast", value=1.2, z_score=0.8),
        ),
        "model_version": "test-0.0.1",
    }
    fields.update(overrides)
    return PredictionResult(**fields)


def test_valid_result_is_accepted() -> None:
    result = make_result()

    assert result.label is TumorLabel.GLIOMA
    assert result.confidence == max(result.probabilities.values())


def test_probabilities_must_sum_to_one() -> None:
    probabilities = {label: 0.3 for label in TumorLabel}

    with pytest.raises(ValueError, match="sum to 1"):
        make_result(probabilities=probabilities, confidence=0.3)


def test_probabilities_must_cover_every_label() -> None:
    probabilities = {TumorLabel.GLIOMA: 0.6, TumorLabel.MENINGIOMA: 0.4}

    with pytest.raises(ValueError, match="one entry per TumorLabel"):
        make_result(probabilities=probabilities, confidence=0.6)


def test_probability_out_of_range_is_rejected() -> None:
    probabilities = {
        TumorLabel.GLIOMA: 1.2,
        TumorLabel.MENINGIOMA: -0.2,
        TumorLabel.PITUITARY: 0.0,
        TumorLabel.NO_TUMOR: 0.0,
    }

    with pytest.raises(ValueError, match="within"):
        make_result(probabilities=probabilities, confidence=1.2)


def test_confidence_must_equal_highest_probability() -> None:
    with pytest.raises(ValueError, match="confidence"):
        make_result(confidence=0.5)


def test_label_must_be_the_argmax() -> None:
    with pytest.raises(ValueError, match="label"):
        make_result(label=TumorLabel.MENINGIOMA)


def test_model_version_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="model_version"):
        make_result(model_version="")


def test_result_is_immutable() -> None:
    result = make_result()

    with pytest.raises(AttributeError):
        result.confidence = 0.1  # type: ignore[misc]


def test_error_types_carry_their_context() -> None:
    metrics = QualityMetrics(snr=2.0, entropy=1.0, skull_stripped=False)

    unsupported = UnsupportedImageError("not an image")
    quality = QualityCheckError("snr too low", metrics)

    assert isinstance(unsupported, MLEngineError)
    assert unsupported.reason == "not an image"
    assert isinstance(quality, MLEngineError)
    assert quality.reason == "snr too low"
    assert quality.metrics == metrics
