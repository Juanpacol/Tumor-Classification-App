import logging
from collections.abc import Callable
from dataclasses import asdict

from anyio import CapacityLimiter, to_thread
from src import errors as ml_errors
from src.contracts import PredictionResult
from src.pipelines import BundleInferencePipeline, InferencePipeline

from domain.errors import (
    ModelUnavailableError,
    QualityCheckFailedError,
    UnsupportedImageError,
)
from domain.models import Prediction

logger = logging.getLogger(__name__)

NOT_LOADED = "the model bundle has not been loaded yet"
NO_BUNDLE_PATH = "MODEL_BUNDLE_PATH not set"

PipelineLoader = Callable[[str], InferencePipeline]


class MLEngineClassifier:
    """ClassifierPort backed by an ml_engine ``InferencePipeline``.

    The pipeline is loaded once by ``load`` (at startup). If loading fails the
    classifier stays unavailable and every ``classify`` raises
    ``ModelUnavailableError``. Inference runs in a worker thread, one at a time.
    """

    def __init__(
        self,
        loader: PipelineLoader = BundleInferencePipeline.from_bundle,
    ) -> None:
        self._loader = loader
        self._pipeline: InferencePipeline | None = None
        self._unavailable_reason = NOT_LOADED
        self._limiter = CapacityLimiter(1)

    async def load(self, bundle_path: str | None) -> None:
        """Load the bundle once; on failure, log the reason and stay unavailable."""
        if not bundle_path:
            self._mark_unavailable(NO_BUNDLE_PATH)
            return
        logger.info("loading model bundle from %s", bundle_path)
        try:
            self._pipeline = await to_thread.run_sync(self._loader, bundle_path)
        except ml_errors.ModelUnavailableError as exc:
            self._mark_unavailable(exc.reason)
        except Exception as exc:
            logger.exception("unexpected error while loading the model bundle")
            self._mark_unavailable(f"unexpected error: {exc}")
        else:
            logger.info("model bundle loaded")

    def is_ready(self) -> bool:
        return self._pipeline is not None

    async def classify(self, study_id: str, image: bytes) -> Prediction:
        """Classify ``image`` off the event loop and map the result to the domain."""
        pipeline = self._pipeline
        if pipeline is None:
            raise ModelUnavailableError(self._unavailable_reason)
        try:
            result = await to_thread.run_sync(
                pipeline.predict, image, limiter=self._limiter
            )
        except ml_errors.UnsupportedImageError as exc:
            raise UnsupportedImageError(exc.reason) from exc
        except ml_errors.QualityCheckError as exc:
            raise QualityCheckFailedError(exc.reason, asdict(exc.metrics)) from exc
        return _to_prediction(study_id, result)

    def _mark_unavailable(self, reason: str) -> None:
        self._pipeline = None
        self._unavailable_reason = reason
        logger.warning("model unavailable: %s", reason)


def _to_prediction(study_id: str, result: PredictionResult) -> Prediction:
    return Prediction(
        study_id=study_id,
        label=result.label.value,
        probabilities={label.value: p for label, p in result.probabilities.items()},
        confidence=result.confidence,
        needs_expert_triage=result.needs_expert_triage,
        quality=asdict(result.quality),
        top_features=[asdict(feature) for feature in result.top_features],
        model_version=result.model_version,
    )
