from typing import Protocol

from src.contracts import PredictionResult


class InferencePipeline(Protocol):
    """Turns an uploaded JPG/PNG, given as bytes, into a ``PredictionResult``.

    Raises ``UnsupportedImageError`` when the input cannot be used and
    ``QualityCheckError`` when it fails the quality gate.
    """

    def predict(self, file_bytes: bytes) -> PredictionResult: ...
