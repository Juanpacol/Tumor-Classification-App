from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adapters.inbound.http.error_handlers import register_exception_handlers
from adapters.inbound.http.router_health import create_health_router
from adapters.outbound.ml.fake_classifier import FakeClassifier
from adapters.outbound.persistence.in_memory_repository import (
    InMemoryStudyRepository,
)
from settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Ether API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    classifier = FakeClassifier()
    repository = InMemoryStudyRepository()
    app.include_router(create_health_router(classifier, repository))

    return app


app = create_app()
