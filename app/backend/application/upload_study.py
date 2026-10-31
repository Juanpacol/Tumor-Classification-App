"""Use case: store a new study image and run it through the classifier."""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from application.list_studies import StudyDetail
from domain.errors import DomainError
from domain.models import Study, StudyStatus
from ports.classifier_port import ClassifierPort
from ports.storage_port import StoragePort
from ports.study_repository_port import StudyRepositoryPort


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
        original_format: str,
        clinical_notes: str | None = None,
    ) -> StudyDetail:
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
        except DomainError as error:
            if error.status_code != 422:
                raise
            study.status = StudyStatus.REJECTED
            study.rejection_reason = error.message
            await self._repository.save(study)
            return StudyDetail(study=study, prediction=None, report=None)

        study.status = StudyStatus.COMPLETED
        await self._repository.save(study)
        await self._repository.save_prediction(prediction)
        return StudyDetail(study=study, prediction=prediction, report=None)
