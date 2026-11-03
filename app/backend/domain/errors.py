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


class FileTooLargeError(DomainError):
    """The uploaded file is larger than the upload limit."""

    def __init__(self, limit_bytes: int):
        super().__init__(
            "FILE_TOO_LARGE",
            f"The file is larger than {limit_bytes // (1024 * 1024)} MB.",
            status_code=413,
        )


class UnsupportedMediaTypeError(DomainError):
    """The uploaded file is not a well-formed JPEG or PNG image."""

    def __init__(self) -> None:
        super().__init__(
            "UNSUPPORTED_MEDIA_TYPE",
            "Only JPG and PNG images are accepted.",
            status_code=415,
        )


class ClassificationFailedError(DomainError):
    """Classification failed unexpectedly; the message stays generic on purpose."""

    def __init__(self) -> None:
        super().__init__(
            "INTERNAL_ERROR",
            "The image could not be processed. Try again later.",
            status_code=500,
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
