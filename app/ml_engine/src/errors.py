"""Errors raised by ml_engine pipelines."""

from src.contracts import QualityMetrics


class MLEngineError(Exception):
    """Base class for every ml_engine error."""


class UnsupportedImageError(MLEngineError):
    """The input cannot be read as a supported MRI image."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class QualityCheckError(MLEngineError):
    """The image was read but failed the quality gate."""

    def __init__(self, reason: str, metrics: QualityMetrics) -> None:
        super().__init__(reason)
        self.reason = reason
        self.metrics = metrics


class ModelUnavailableError(MLEngineError):
    """A model bundle is missing, corrupt or does not match the configuration."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
