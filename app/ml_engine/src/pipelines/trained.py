"""Serving pipeline: image bytes in, ``PredictionResult`` out."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.classical.extractor import extract_radiomics
from src.contracts import TRIAGE_CONFIDENCE_THRESHOLD, PredictionResult, TumorLabel
from src.deep_learning.extractor import DeepFeatureExtractor
from src.pipelines.artifact import LoadedArtifact, load_artifact
from src.pipelines.top_features import top_features
from src.preprocessing.loading import decode_mri_image
from src.preprocessing.prepare import prepare_image


class BundleInferencePipeline:
    """Runs the exact preprocessing, features and classifier the artifact was trained with."""

    def __init__(
        self,
        artifact: LoadedArtifact,
        extractor: DeepFeatureExtractor,
        triage_threshold: float = TRIAGE_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._artifact = artifact
        self._extractor = extractor
        self._triage_threshold = triage_threshold

    @classmethod
    def from_bundle(
        cls,
        directory: str | Path,
        triage_threshold: float = TRIAGE_CONFIDENCE_THRESHOLD,
    ) -> BundleInferencePipeline:
        """Load and verify every artifact once.

        Raises ``ModelUnavailableError`` if the folder is incomplete, altered or
        inconsistent, so the caller can fail fast at start-up.
        """
        directory = Path(directory)
        artifact = load_artifact(directory)
        extractor = DeepFeatureExtractor.from_bundle(directory, artifact.preprocessing)
        return cls(artifact, extractor, triage_threshold)

    def predict(self, file_bytes: bytes) -> PredictionResult:
        """Classify a JPG/PNG given as bytes.

        Raises ``UnsupportedImageError`` or ``QualityCheckError`` for images that
        cannot be used.
        """
        artifact = self._artifact
        prepared = prepare_image(decode_mri_image(file_bytes), artifact.preprocessing)
        radiomic = extract_radiomics(
            prepared.radiomics_image, prepared.mask, artifact.radiomics
        ).values.astype(np.float32)
        embedding = self._extractor.extract(prepared.cnn_image)

        row = np.hstack([radiomic, embedding])[None]
        scores = artifact.classifier.predict_proba(row)[0]
        probabilities = {
            TumorLabel(label): float(score)
            for label, score in zip(artifact.classifier.classes_, scores)
        }
        label = max(probabilities, key=probabilities.__getitem__)
        confidence = probabilities[label]
        return PredictionResult(
            label=label,
            probabilities=probabilities,
            confidence=confidence,
            needs_expert_triage=confidence < self._triage_threshold,
            quality=prepared.quality,
            top_features=top_features(radiomic, artifact.stats),
            model_version=artifact.version,
        )
