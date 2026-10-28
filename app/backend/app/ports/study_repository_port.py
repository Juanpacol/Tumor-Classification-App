from typing import Protocol

from app.domain.models import Study


class StudyRepositoryPort(Protocol):
    def get(self, study_id: str) -> Study | None: ...

    def save(self, study: Study) -> None: ...

    def ping(self) -> bool: ...
