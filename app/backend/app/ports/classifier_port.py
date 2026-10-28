from typing import Protocol

from app.domain.models import Prediction


class ClassifierPort(Protocol):
    def classify(self, image_path: str) -> Prediction: ...

    def is_ready(self) -> bool: ...
