"""Use case: store a new study image and run it through the classifier."""

import logging
import traceback
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from anyio import to_thread

from application.image_upload import prepare_upload
from application.list_studies import StudyDetail
from domain.errors import (
    ClassificationFailedError,
    DomainError,
    ModelUnavailableError,
    QualityCheckFailedError,
    UnsupportedImageError,
)
from domain.models import Study, StudyStatus
from ports.classifier_port import ClassifierPort
from ports.storage_port import StoragePort
from ports.study_repository_port import StudyRepositoryPort

logger = logging.getLogger(__name__)

REJECTIONS = (QualityCheckFailedError, UnsupportedImageError)


class StudyRejectedError(DomainError):
    """The classifier rejected the image; carries the study saved as REJECTED."""

    def __init__(self, detail: StudyDetail, cause: DomainError) -> None:
        super().__init__(cause.code, cause.message, cause.status_code, cause.details)
        self.detail = detail


class UploadStudy:
    def __init__(
        self,
        repository: StudyRepositoryPort,
        storage: StoragePort,
        classifier: ClassifierPort,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._classifier = classifier
        self._clock = clock

    async def execute(
        self,
        image: bytes,
        clinical_notes: str | None = None,
    ) -> StudyDetail:
        """Store the image without its metadata, classify it and save the outcome.

        Nothing is stored when no model is loaded or when the file is too large
        or not a decodable JPG/PNG. A rejected image is saved as REJECTED with its
        reason and raised as ``StudyRejectedError``. Any other classifier error
        is saved as FAILED; an unexpected one is logged without its message and
        raised as ``ClassificationFailedError``.
        """
        if not self._classifier.is_ready():
            raise ModelUnavailableError("no model is loaded")
        image, original_format = await to_thread.run_sync(prepare_upload, image)
        study = Study(
            id=str(uuid.uuid4()),
            file_key="",
            original_format=original_format,
            status=StudyStatus.PENDING,
            created_at=self._clock(),
            clinical_notes=clinical_notes,
        )
        study.file_key = self._storage.save(image, f"studies/{study.id}")
        await self._repository.save(study)

        try:
            prediction = await self._classifier.classify(study.id, image)
        except REJECTIONS as error:
            raise await self._reject(study, error) from None
        except DomainError:
            await self._save_status(study, StudyStatus.FAILED)
            raise
        except Exception as error:  # noqa: BLE001 - no exc_info: it may carry PHI
            await self._save_status(study, StudyStatus.FAILED)
            _log_failure(study.id, error)
            raise ClassificationFailedError() from None

        await self._save_status(study, StudyStatus.COMPLETED)
        await self._repository.save_prediction(prediction)
        return StudyDetail(study=study, prediction=prediction, report=None)

    async def _reject(self, study: Study, error: DomainError) -> StudyRejectedError:
        study.rejection_reason = (error.details or {}).get("reason", error.message)
        await self._save_status(study, StudyStatus.REJECTED)
        detail = StudyDetail(study=study, prediction=None, report=None)
        return StudyRejectedError(detail, error)

    async def _save_status(self, study: Study, status: StudyStatus) -> None:
        study.status = status
        await self._repository.save(study)


def _log_failure(study_id: str, error: Exception) -> None:
    """Log the failure by study id, error type and raising frame, never its message.

    The message may echo patient data, so neither it nor the traceback is logged.
    """
    frames = traceback.extract_tb(error.__traceback__)
    where = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
    logger.error(
        "study %s classification failed: %s at %s",
        study_id,
        type(error).__name__,
        where,
    )
