import hashlib
from pathlib import Path

import numpy as np

from src.contracts import (
    TRIAGE_CONFIDENCE_THRESHOLD,
    FeatureContribution,
    PredictionResult,
    QualityMetrics,
    TumorLabel,
)
from src.errors import UnsupportedImageError

FAKE_MODEL_VERSION = "fake-0.1.0"
_FEATURE_NAMES = ("glcm_contrast", "lbp_uniformity", "wavelet_energy_lh")


class FakeInferencePipeline:
    """Deterministic stand-in so consumers can integrate before a model exists.

    The result depends only on the bytes of the image: the same file always
    yields the same prediction.
    """

    def predict(self, image_path: str | Path) -> PredictionResult:
        path = Path(image_path)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise UnsupportedImageError(f"cannot read image: {path}") from exc

        digest = hashlib.sha256(content).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))

        logits = rng.normal(size=len(TumorLabel))
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        probabilities = {label: float(p) for label, p in zip(TumorLabel, probs)}
        label = max(probabilities, key=probabilities.__getitem__)
        confidence = probabilities[label]

        return PredictionResult(
            label=label,
            probabilities=probabilities,
            confidence=confidence,
            needs_expert_triage=confidence < TRIAGE_CONFIDENCE_THRESHOLD,
            quality=QualityMetrics(
                snr=float(rng.uniform(10.0, 30.0)),
                entropy=float(rng.uniform(4.0, 7.5)),
                skull_stripped=True,
            ),
            top_features=tuple(
                FeatureContribution(
                    name=name,
                    value=float(rng.normal()),
                    z_score=float(rng.normal()),
                )
                for name in _FEATURE_NAMES
            ),
            model_version=FAKE_MODEL_VERSION,
        )
