import math
from pathlib import Path

import pytest

from src.contracts import PROBABILITY_TOLERANCE, TumorLabel
from src.errors import UnsupportedImageError
from src.pipelines import FakeInferencePipeline, InferencePipeline


def write_image(directory: Path, name: str, content: bytes) -> Path:
    path = directory / name
    path.write_bytes(content)
    return path


def test_fake_pipeline_satisfies_the_protocol(tmp_path: Path) -> None:
    pipeline: InferencePipeline = FakeInferencePipeline()

    result = pipeline.predict(write_image(tmp_path, "a.png", b"scan-a"))

    assert result.model_version == "fake-0.1.0"


def test_same_image_gives_same_result(tmp_path: Path) -> None:
    image = write_image(tmp_path, "a.png", b"scan-a")
    pipeline = FakeInferencePipeline()

    assert pipeline.predict(image) == pipeline.predict(image)


def test_result_depends_on_content_not_path(tmp_path: Path) -> None:
    first = write_image(tmp_path, "one.png", b"scan-a")
    copy = write_image(tmp_path, "two.png", b"scan-a")
    pipeline = FakeInferencePipeline()

    assert pipeline.predict(first) == pipeline.predict(copy)


def test_different_images_give_different_results(tmp_path: Path) -> None:
    pipeline = FakeInferencePipeline()

    first = pipeline.predict(write_image(tmp_path, "a.png", b"scan-a"))
    second = pipeline.predict(write_image(tmp_path, "b.png", b"scan-b"))

    assert first.probabilities != second.probabilities


def test_contract_invariants_hold_across_many_images(tmp_path: Path) -> None:
    pipeline = FakeInferencePipeline()

    for index in range(50):
        result = pipeline.predict(
            write_image(tmp_path, f"{index}.png", f"scan-{index}".encode())
        )

        probabilities = result.probabilities
        assert set(probabilities) == set(TumorLabel)
        assert math.isclose(
            sum(probabilities.values()), 1.0, abs_tol=PROBABILITY_TOLERANCE
        )
        assert result.confidence == max(probabilities.values())
        assert result.label == max(probabilities, key=probabilities.__getitem__)
        assert result.needs_expert_triage == (result.confidence < 0.7)


def test_missing_file_raises_unsupported_image_error(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedImageError, match="cannot read image"):
        FakeInferencePipeline().predict(tmp_path / "missing.png")
