from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adapters.inbound.http.error_handlers import register_exception_handlers
from adapters.inbound.http.router_health import create_health_router
from adapters.outbound.ml.fake_classifier import FakeClassifier
from adapters.outbound.persistence.sqlalchemy.engine import (
    create_engine,
    create_session_factory,
)
from adapters.outbound.persistence.sqlalchemy.repository import SqlAlchemyStudyRepository
from ports.study_repository_port import StudyRepositoryPort
from settings import get_settings


def create_app(repository: StudyRepositoryPort | None = None) -> FastAPI:
    """Build the API; pass a repository to bypass the PostgreSQL adapter (tests)."""
    settings = get_settings()
    engine = None
    if repository is None:
        engine = create_engine(settings)
        repository = SqlAlchemyStudyRepository(create_session_factory(engine))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if engine is not None:
            await engine.dispose()

    app = FastAPI(
        title="Ether API",
        version="0.1.0",
        description=("Clinical decision support API for brain tumor detection on MRI scans."),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    classifier = FakeClassifier()
    app.include_router(create_health_router(classifier, repository))

    return app


app = create_app()
