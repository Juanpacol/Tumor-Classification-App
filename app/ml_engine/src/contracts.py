"""Typed inference contract between ml_engine and its consumers."""

import math
from dataclasses import dataclass
from enum import Enum

PROBABILITY_TOLERANCE = 1e-6
TRIAGE_CONFIDENCE_THRESHOLD = 0.7


class TumorLabel(str, Enum):
    """Classes the classifier can predict."""

    GLIOMA = "glioma"
    MENINGIOMA = "meningioma"
    PITUITARY = "pituitary"
    NO_TUMOR = "no_tumor"


@dataclass(frozen=True)
class QualityMetrics:
    """Image quality measurements taken before classification."""

    snr: float
    entropy: float
    skull_stripped: bool


@dataclass(frozen=True)
class FeatureContribution:
    """A radiomic feature that drove the prediction."""

    name: str
    value: float
    z_score: float


@dataclass(frozen=True)
class PredictionResult:
    """Output of an inference pipeline.

    Invariants are checked on construction: probabilities cover every
    ``TumorLabel`` and sum to 1, ``confidence`` equals the highest
    probability and ``label`` is the class holding it.
    """

    label: TumorLabel
    probabilities: dict[TumorLabel, float]
    confidence: float
    needs_expert_triage: bool
    quality: QualityMetrics
    top_features: tuple[FeatureContribution, ...]
    model_version: str

    def __post_init__(self) -> None:
        if set(self.probabilities) != set(TumorLabel):
            raise ValueError(
                "probabilities must contain exactly one entry per TumorLabel"
            )
        if any(not 0.0 <= p <= 1.0 for p in self.probabilities.values()):
            raise ValueError("probabilities must be within [0, 1]")
        total = sum(self.probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=PROBABILITY_TOLERANCE):
            raise ValueError(f"probabilities must sum to 1, got {total}")
        highest = max(self.probabilities.values())
        if not math.isclose(self.confidence, highest, abs_tol=PROBABILITY_TOLERANCE):
            raise ValueError("confidence must equal the highest probability")
        if self.probabilities[self.label] != highest:
            raise ValueError("label must be the class with the highest probability")
        if not self.model_version:
            raise ValueError("model_version must not be empty")
