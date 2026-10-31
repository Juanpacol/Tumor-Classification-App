from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adapters.inbound.http.error_handlers import register_exception_handlers
from adapters.inbound.http.router_health import create_health_router
from adapters.inbound.http.router_studies import create_studies_router
from adapters.outbound.llm.factory import build_llm
from adapters.outbound.ml.ml_engine_classifier import MLEngineClassifier
from adapters.outbound.persistence.sqlalchemy.engine import (
    create_engine,
    create_session_factory,
)
from adapters.outbound.persistence.sqlalchemy.repository import (
    SqlAlchemyStudyRepository,
)
from adapters.outbound.storage.filesystem_storage import FilesystemStorage
from application.get_study import GetStudy
from application.list_studies import ListStudies
from application.upload_study import UploadStudy
from ports.classifier_port import ClassifierPort
from ports.llm_port import LLMPort
from ports.storage_port import StoragePort
from ports.study_repository_port import StudyRepositoryPort
from settings import get_settings


def create_app(
    repository: StudyRepositoryPort | None = None,
    llm: LLMPort | None = None,
    classifier: ClassifierPort | None = None,
    storage: StoragePort | None = None,
) -> FastAPI:
    """Build the API; pass a repository, an LLM or a classifier to replace the real
    ones (tests). An ``MLEngineClassifier`` loads its bundle once, at startup."""
    settings = get_settings()
    llm = llm or build_llm(settings)
    classifier = classifier or MLEngineClassifier()
    storage = storage or FilesystemStorage(settings.storage_dir)
    engine = None
    if repository is None:
        engine = create_engine(settings)
        repository = SqlAlchemyStudyRepository(create_session_factory(engine))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if isinstance(classifier, MLEngineClassifier):
            await classifier.load(settings.model_bundle_path)
        yield
        await llm.aclose()
        if engine is not None:
            await engine.dispose()

    app = FastAPI(
        title="Ether API",
        version="0.1.0",
        description=(
            "Clinical decision support API for brain tumor detection on MRI scans."
        ),
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
    app.state.llm = llm

    app.include_router(create_health_router(classifier, repository))
    app.include_router(
        create_studies_router(
            repository,
            storage,
            ListStudies(repository),
            GetStudy(repository),
            UploadStudy(repository, storage, classifier),
        )
    )

    return app


app = create_app()
