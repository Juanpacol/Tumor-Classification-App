from pathlib import Path
from typing import Protocol

from src.contracts import PredictionResult


class InferencePipeline(Protocol):
    """Turns an MRI image into a ``PredictionResult``.

    Raises ``UnsupportedImageError`` when the input cannot be read and
    ``QualityCheckError`` when it fails the quality gate.
    """

    def predict(self, image_path: str | Path) -> PredictionResult: ...
