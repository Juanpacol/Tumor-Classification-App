import math
import os
from pathlib import Path

import pytest

from adapters.outbound.ml.ml_engine_classifier import MLEngineClassifier
from settings import Settings

BUNDLE_PATH = Settings().model_bundle_path
SAMPLE_IMAGE = os.environ.get("ML_SAMPLE_IMAGE")

pytestmark = pytest.mark.skipif(
    not BUNDLE_PATH or not (Path(BUNDLE_PATH) / "manifest.json").is_file(),
    reason=f"no model bundle (manifest.json) at MODEL_BUNDLE_PATH ({BUNDLE_PATH!r})",
)


@pytest.fixture
async def classifier() -> MLEngineClassifier:
    classifier = MLEngineClassifier()
    await classifier.load(BUNDLE_PATH)
    return classifier


async def test_real_bundle_loads(classifier: MLEngineClassifier) -> None:
    assert classifier.is_ready()


@pytest.mark.skipif(
    not SAMPLE_IMAGE or not Path(SAMPLE_IMAGE).is_file(),
    reason=f"no sample image at ML_SAMPLE_IMAGE ({SAMPLE_IMAGE!r})",
)
async def test_real_bundle_classifies_a_sample_image(
    classifier: MLEngineClassifier,
) -> None:
    image = Path(str(SAMPLE_IMAGE)).read_bytes()

    prediction = await classifier.classify("study-1", image)

    assert prediction.study_id == "study-1"
    assert prediction.label in prediction.probabilities
    assert math.isclose(sum(prediction.probabilities.values()), 1.0, abs_tol=1e-6)
    assert prediction.model_version
