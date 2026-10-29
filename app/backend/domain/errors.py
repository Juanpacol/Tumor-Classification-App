from typing import Any


class DomainError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


class ModelUnavailableError(DomainError):
    """No classifier model is loaded, so images cannot be classified."""

    def __init__(self, reason: str):
        super().__init__(
            "MODEL_UNAVAILABLE",
            f"The classification model is not available: {reason}",
            status_code=503,
        )


class UnsupportedImageError(DomainError):
    """The uploaded file cannot be read as a supported MRI image."""

    def __init__(self, reason: str):
        super().__init__(
            "UNSUPPORTED_IMAGE",
            "The image cannot be used for classification.",
            status_code=422,
            details={"reason": reason},
        )


class QualityCheckFailedError(DomainError):
    """The image was read but failed the quality gate, with the measured metrics."""

    def __init__(self, reason: str, metrics: dict[str, Any]):
        super().__init__(
            "QUALITY_CHECK_FAILED",
            "The image did not pass the quality check.",
            status_code=422,
            details={"reason": reason, **metrics},
        )
