from typing import Protocol

from domain.models import Prediction


class ClassifierPort(Protocol):
    async def classify(self, study_id: str, image: bytes) -> Prediction: ...

    def is_ready(self) -> bool: ...
