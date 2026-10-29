import math

import pytest

from src.contracts import (
    PROBABILITY_TOLERANCE,
    TRIAGE_CONFIDENCE_THRESHOLD,
    TumorLabel,
)
from src.errors import UnsupportedImageError
from src.pipelines import FakeInferencePipeline, InferencePipeline


def test_fake_pipeline_satisfies_the_protocol() -> None:
    pipeline: InferencePipeline = FakeInferencePipeline()

    assert pipeline.predict(b"scan-a").model_version == "fake-0.1.0"


def test_same_input_gives_same_result() -> None:
    pipeline = FakeInferencePipeline()

    assert pipeline.predict(b"scan-a") == pipeline.predict(b"scan-a")


def test_different_inputs_give_different_results() -> None:
    pipeline = FakeInferencePipeline()

    first = pipeline.predict(b"scan-a")
    second = pipeline.predict(b"scan-b")

    assert first.probabilities != second.probabilities


def test_contract_invariants_hold_across_many_inputs() -> None:
    pipeline = FakeInferencePipeline()

    for index in range(50):
        result = pipeline.predict(f"scan-{index}".encode())

        probabilities = result.probabilities
        assert set(probabilities) == set(TumorLabel)
        assert math.isclose(
            sum(probabilities.values()), 1.0, abs_tol=PROBABILITY_TOLERANCE
        )
        assert result.confidence == max(probabilities.values())
        assert result.label == max(probabilities, key=probabilities.__getitem__)
        assert result.needs_expert_triage == (
            result.confidence < TRIAGE_CONFIDENCE_THRESHOLD
        )


def test_empty_input_raises_unsupported_image_error() -> None:
    with pytest.raises(UnsupportedImageError, match="empty input"):
        FakeInferencePipeline().predict(b"")
