"""Use case: fetch one study with its prediction and report."""

from application.list_studies import StudyDetail
from domain.errors import DomainError
from ports.study_repository_port import StudyRepositoryPort


class GetStudy:
    def __init__(self, repository: StudyRepositoryPort) -> None:
        self._repository = repository

    async def execute(self, study_id: str) -> StudyDetail:
        study = await self._repository.get(study_id)
        if study is None:
            raise DomainError("STUDY_NOT_FOUND", "Study not found.", status_code=404)
        prediction = await self._repository.get_prediction(study_id)
        report = await self._repository.get_report(study_id)
        return StudyDetail(study=study, prediction=prediction, report=report)
