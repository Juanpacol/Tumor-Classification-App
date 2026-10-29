from fastapi import APIRouter

from ports.classifier_port import ClassifierPort
from ports.study_repository_port import StudyRepositoryPort


def create_health_router(
    classifier: ClassifierPort,
    repository: StudyRepositoryPort,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, bool | str]:
        return {
            "status": "ok",
            "model_loaded": classifier.is_ready(),
            "database_reachable": repository.ping(),
        }

    return router
