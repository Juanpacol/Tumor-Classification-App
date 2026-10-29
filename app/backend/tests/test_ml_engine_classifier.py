import asyncio
import logging
import math
import threading

import pytest
from src import errors as ml_errors
from src.contracts import PredictionResult, QualityMetrics
from src.pipelines import FakeInferencePipeline

from adapters.outbound.ml.ml_engine_classifier import MLEngineClassifier
from domain.errors import (
    ModelUnavailableError,
    QualityCheckFailedError,
    UnsupportedImageError,
)
from tests.doubles import RecordingPipeline

METRICS = QualityMetrics(snr=1.5, entropy=0.8, skull_stripped=False)


class FailingQualityPipeline:
    def predict(self, file_bytes: bytes) -> PredictionResult:
        raise ml_errors.QualityCheckError("snr below threshold", METRICS)


async def loaded(pipeline: object) -> MLEngineClassifier:
    classifier = MLEngineClassifier(loader=lambda _: pipeline)
    await classifier.load("/models/current")
    return classifier


async def test_classifier_is_not_ready_before_load() -> None:
    classifier = MLEngineClassifier(loader=lambda _: FakeInferencePipeline())

    assert not classifier.is_ready()
    with pytest.raises(ModelUnavailableError):
        await classifier.classify("study-1", b"scan")


async def test_unset_bundle_path_leaves_model_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loads: list[str] = []
    classifier = MLEngineClassifier(loader=lambda path: loads.append(path))

    with caplog.at_level(logging.WARNING):
        await classifier.load(None)

    assert not classifier.is_ready()
    assert loads == []
    assert "MODEL_BUNDLE_PATH not set" in caplog.text
    with pytest.raises(ModelUnavailableError, match="MODEL_BUNDLE_PATH not set"):
        await classifier.classify("study-1", b"scan")


async def test_invalid_bundle_leaves_model_unavailable_with_its_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def reject(path: str) -> FakeInferencePipeline:
        raise ml_errors.ModelUnavailableError(f"artifact manifest not found: {path}")

    classifier = MLEngineClassifier(loader=reject)
    with caplog.at_level(logging.WARNING):
        await classifier.load("/models/current")

    assert not classifier.is_ready()
    assert "artifact manifest not found: /models/current" in caplog.text
    with pytest.raises(ModelUnavailableError) as raised:
        await classifier.classify("study-1", b"scan")
    assert raised.value.status_code == 503


async def test_unexpected_load_error_leaves_model_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def explode(path: str) -> FakeInferencePipeline:
        raise RuntimeError("disk on fire")

    classifier = MLEngineClassifier(loader=explode)
    with caplog.at_level(logging.WARNING):
        await classifier.load("/models/current")

    assert not classifier.is_ready()
    assert "disk on fire" in caplog.text


async def test_classify_maps_the_result_to_a_domain_prediction() -> None:
    classifier = await loaded(FakeInferencePipeline())
    expected = FakeInferencePipeline().predict(b"scan")

    prediction = await classifier.classify("study-1", b"scan")

    assert classifier.is_ready()
    assert prediction.study_id == "study-1"
    assert prediction.label == expected.label.value
    assert set(prediction.probabilities) == {
        "glioma",
        "meningioma",
        "pituitary",
        "no_tumor",
    }
    assert math.isclose(sum(prediction.probabilities.values()), 1.0, abs_tol=1e-6)
    assert prediction.confidence == expected.confidence
    assert prediction.needs_expert_triage == expected.needs_expert_triage
    assert set(prediction.quality) == {"snr", "entropy", "skull_stripped"}
    assert [f["name"] for f in prediction.top_features] == [
        f.name for f in expected.top_features
    ]
    assert set(prediction.top_features[0]) == {"name", "value", "z_score"}
    assert prediction.model_version == "fake-0.1.0"


async def test_unsupported_image_maps_to_domain_error_with_reason() -> None:
    classifier = await loaded(FakeInferencePipeline())

    with pytest.raises(UnsupportedImageError) as raised:
        await classifier.classify("study-1", b"")

    assert raised.value.details == {"reason": "empty input"}


async def test_quality_failure_maps_to_domain_error_with_metrics() -> None:
    classifier = await loaded(FailingQualityPipeline())

    with pytest.raises(QualityCheckFailedError) as raised:
        await classifier.classify("study-1", b"scan")

    assert raised.value.details == {
        "reason": "snr below threshold",
        "snr": 1.5,
        "entropy": 0.8,
        "skull_stripped": False,
    }


async def test_predict_runs_off_the_event_loop_thread() -> None:
    pipeline = RecordingPipeline()
    classifier = await loaded(pipeline)

    await classifier.classify("study-1", b"scan")

    assert pipeline.threads and pipeline.threads[0] != threading.get_ident()


async def test_concurrent_classifications_never_overlap() -> None:
    pipeline = RecordingPipeline(delay=0.05)
    classifier = await loaded(pipeline)

    first, second = await asyncio.gather(
        classifier.classify("study-1", b"scan-a"),
        classifier.classify("study-2", b"scan-b"),
    )

    assert pipeline.max_active == 1
    assert (first.study_id, second.study_id) == ("study-1", "study-2")
