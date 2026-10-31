from domain.errors import DomainError
from domain.models import ChatMessage, Prediction, Report, Study


class InMemoryStudyRepository:
    def __init__(self) -> None:
        self._studies: dict[str, Study] = {}
        self._predictions: dict[str, Prediction] = {}
        self._reports: dict[str, Report] = {}
        self._messages: dict[str, list[ChatMessage]] = {}

    async def get(self, study_id: str) -> Study | None:
        return self._studies.get(study_id)

    async def save(self, study: Study) -> None:
        self._studies[study.id] = study

    async def list_recent(self, page: int, page_size: int) -> tuple[list[Study], int]:
        studies = sorted(
            self._studies.values(), key=lambda s: s.created_at, reverse=True
        )
        start = (page - 1) * page_size
        return studies[start : start + page_size], len(studies)

    async def ping(self) -> bool:
        return True

    async def save_prediction(self, prediction: Prediction) -> None:
        self._require_study(prediction.study_id)
        self._predictions[prediction.study_id] = prediction

    async def get_prediction(self, study_id: str) -> Prediction | None:
        return self._predictions.get(study_id)

    async def save_report(self, report: Report) -> None:
        self._require_study(report.study_id)
        self._reports[report.study_id] = report

    async def get_report(self, study_id: str) -> Report | None:
        return self._reports.get(study_id)

    async def add_message(self, message: ChatMessage) -> None:
        self._require_study(message.study_id)
        self._messages.setdefault(message.study_id, []).append(message)

    async def list_messages(self, study_id: str) -> list[ChatMessage]:
        return sorted(self._messages.get(study_id, []), key=lambda m: m.created_at)

    def _require_study(self, study_id: str) -> None:
        if study_id not in self._studies:
            raise DomainError("STUDY_NOT_FOUND", "Study does not exist")
