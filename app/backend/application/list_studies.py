"""Use case: list the most recent studies, with their prediction and report."""

from dataclasses import dataclass

from domain.models import Prediction, Report, Study
from ports.study_repository_port import StudyRepositoryPort


@dataclass
class StudyDetail:
    study: Study
    prediction: Prediction | None
    report: Report | None


@dataclass
class StudyPage:
    items: list[StudyDetail]
    page: int
    page_size: int
    total: int


class ListStudies:
    def __init__(self, repository: StudyRepositoryPort) -> None:
        self._repository = repository

    async def execute(self, page: int = 1, page_size: int = 20) -> StudyPage:
        studies, total = await self._repository.list_recent(page, page_size)
        items = [await self._with_results(study) for study in studies]
        return StudyPage(items=items, page=page, page_size=page_size, total=total)

    async def _with_results(self, study: Study) -> StudyDetail:
        prediction = await self._repository.get_prediction(study.id)
        report = await self._repository.get_report(study.id)
        return StudyDetail(study=study, prediction=prediction, report=report)
