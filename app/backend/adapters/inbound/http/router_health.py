from fastapi import APIRouter
from pydantic import BaseModel

from ports.classifier_port import ClassifierPort
from ports.study_repository_port import StudyRepositoryPort


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_reachable: bool


def create_health_router(
    classifier: ClassifierPort,
    repository: StudyRepositoryPort,
) -> APIRouter:
    router = APIRouter(tags=["Health"])

    @router.get(
        "/health",
        response_model=HealthResponse,
        summary="Check service health",
        description=(
            "Reports whether the API is up, the classifier model is loaded "
            "and the study repository is reachable."
        ),
    )
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=classifier.is_ready(),
            database_reachable=repository.ping(),
        )

    return router
