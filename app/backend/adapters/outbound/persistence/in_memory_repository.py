from domain.models import Study


class InMemoryStudyRepository:
    def __init__(self) -> None:
        self._studies: dict[str, Study] = {}

    def get(self, study_id: str) -> Study | None:
        return self._studies.get(study_id)

    def save(self, study: Study) -> None:
        self._studies[study.id] = study

    def ping(self) -> bool:
        return True
