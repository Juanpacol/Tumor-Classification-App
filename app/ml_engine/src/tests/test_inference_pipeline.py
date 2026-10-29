import io
import time
from pathlib import Path

import joblib
import numpy as np
import pytest
from PIL import Image

from src.classical.config import RadiomicsConfig
from src.contracts import PredictionResult
from src.deep_learning.extractor import DeepFeatureExtractor
from src.errors import QualityCheckError, UnsupportedImageError
from src.pipelines import BundleInferencePipeline, InferencePipeline
from src.preprocessing.prepare import PreprocessingConfig
from src.training.artifact import CLASSIFIER_FILE
from src.training.features import make_featurizer
from src.training.run import TrainingResult

SAMPLES = [
    "Test/glioma/0.png",
    "Test/meningioma/3.png",
    "Test/notumor/7.png",
    "Test/pituitary/11.png",
    "Test/glioma/15.png",
]


@pytest.fixture(scope="module")
def pipeline(result: TrainingResult) -> BundleInferencePipeline:
    return BundleInferencePipeline.from_bundle(result.artifact_dir)


def image_bytes(workspace: dict[str, Path], relative: str) -> bytes:
    return (workspace["data"] / relative).read_bytes()


def png_bytes(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def test_pipeline_satisfies_the_protocol(
    result: TrainingResult, workspace: dict[str, Path]
) -> None:
    served: InferencePipeline = BundleInferencePipeline.from_bundle(result.artifact_dir)

    assert served.predict(image_bytes(workspace, SAMPLES[0])).model_version == "9.9.9"


def test_prediction_is_a_valid_result_with_the_artifact_version(
    pipeline: BundleInferencePipeline, workspace: dict[str, Path]
) -> None:
    prediction = pipeline.predict(image_bytes(workspace, SAMPLES[0]))

    assert isinstance(prediction, PredictionResult)
    assert prediction.model_version == "9.9.9"
    assert prediction.confidence == max(prediction.probabilities.values())
    assert prediction.quality.snr > 0 and prediction.quality.entropy > 0


def test_top_features_are_the_five_largest_z_scores(
    pipeline: BundleInferencePipeline, workspace: dict[str, Path]
) -> None:
    features = pipeline.predict(image_bytes(workspace, SAMPLES[1])).top_features

    assert len(features) == 5
    scores = [abs(f.z_score) for f in features]
    assert scores == sorted(scores, reverse=True)
    assert len({f.name for f in features}) == 5


@pytest.mark.parametrize("relative", SAMPLES)
def test_serving_matches_training_for_the_same_image(
    pipeline: BundleInferencePipeline,
    result: TrainingResult,
    workspace: dict[str, Path],
    relative: str,
) -> None:
    """No train/serve skew: the same image gives the same probabilities both ways."""
    classifier = joblib.load(result.artifact_dir / CLASSIFIER_FILE)
    extractor = DeepFeatureExtractor.from_bundle(result.artifact_dir)
    featurize, _ = make_featurizer(extractor, PreprocessingConfig(), RadiomicsConfig())
    radiomics, cnn = featurize(workspace["data"] / relative)
    expected = classifier.predict_proba(np.hstack([radiomics, cnn])[None])[0]

    served = pipeline.predict(image_bytes(workspace, relative))

    by_class = {label.value: p for label, p in served.probabilities.items()}
    assert [by_class[name] for name in classifier.classes_] == pytest.approx(
        list(expected), abs=1e-6
    )
    assert served.label.value == classifier.classes_[int(np.argmax(expected))]


def test_default_triage_threshold_is_0_75(
    pipeline: BundleInferencePipeline, workspace: dict[str, Path]
) -> None:
    prediction = pipeline.predict(image_bytes(workspace, SAMPLES[2]))

    assert prediction.needs_expert_triage == (prediction.confidence < 0.75)


@pytest.mark.parametrize(("threshold", "triage"), [(1.01, True), (0.0, False)])
def test_triage_threshold_is_configurable(
    result: TrainingResult,
    workspace: dict[str, Path],
    threshold: float,
    triage: bool,
) -> None:
    custom = BundleInferencePipeline.from_bundle(
        result.artifact_dir, triage_threshold=threshold
    )

    assert (
        custom.predict(image_bytes(workspace, SAMPLES[3])).needs_expert_triage is triage
    )


def test_garbage_bytes_are_an_unsupported_image(
    pipeline: BundleInferencePipeline,
) -> None:
    with pytest.raises(UnsupportedImageError):
        pipeline.predict(b"not an image")


def test_a_blank_image_fails_the_quality_gate_with_its_metrics(
    pipeline: BundleInferencePipeline,
) -> None:
    blank = png_bytes(np.zeros((128, 128), dtype=np.uint8))

    with pytest.raises(QualityCheckError) as error:
        pipeline.predict(blank)

    assert error.value.metrics.entropy == 0.0


def test_one_image_takes_under_three_seconds_once_loaded(
    pipeline: BundleInferencePipeline, workspace: dict[str, Path]
) -> None:
    data = image_bytes(workspace, SAMPLES[4])
    pipeline.predict(data)

    start = time.perf_counter()
    pipeline.predict(data)

    assert time.perf_counter() - start < 3.0
