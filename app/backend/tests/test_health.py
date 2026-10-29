import time
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from src import errors as ml_errors
from src.pipelines import FakeInferencePipeline, InferencePipeline

from adapters.outbound.ml.ml_engine_classifier import MLEngineClassifier
from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from main import create_app
from settings import get_settings
from tests.doubles import RecordingPipeline

BUNDLE_PATH = "/models/current"


class UnreachableRepository(InMemoryStudyRepository):
    async def ping(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def bundle_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("MODEL_BUNDLE_PATH", BUNDLE_PATH)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def client_with(
    loader: Callable[[str], InferencePipeline],
    repository: InMemoryStudyRepository | None = None,
) -> tuple[TestClient, MLEngineClassifier]:
    classifier = MLEngineClassifier(loader=loader)
    app = create_app(
        repository=repository or InMemoryStudyRepository(), classifier=classifier
    )
    return TestClient(app), classifier


def test_health_returns_ok_with_statuses():
    client, _ = client_with(lambda _: FakeInferencePipeline())

    with client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["database_reachable"] is True


def test_health_succeeds_when_database_unreachable():
    client, _ = client_with(
        lambda _: FakeInferencePipeline(), repository=UnreachableRepository()
    )

    with client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["database_reachable"] is False


def test_health_reports_model_unavailable_without_bundle_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("MODEL_BUNDLE_PATH")
    get_settings.cache_clear()
    client, _ = client_with(lambda _: FakeInferencePipeline())

    with client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_health_reports_model_unavailable_with_invalid_bundle():
    def reject(path: str) -> InferencePipeline:
        raise ml_errors.ModelUnavailableError("invalid artifact manifest")

    client, _ = client_with(reject)

    with client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_bundle_is_loaded_once_at_startup():
    loads: list[str] = []

    def counting_loader(path: str) -> InferencePipeline:
        loads.append(path)
        return FakeInferencePipeline()

    client, classifier = client_with(counting_loader)

    with client:
        assert loads == [BUNDLE_PATH]
        for index in range(3):
            client.get("/health")
            client.portal.call(classifier.classify, f"study-{index}", b"scan")

    assert loads == [BUNDLE_PATH]


def test_health_answers_while_an_inference_is_running():
    pipeline = RecordingPipeline(delay=1.0)
    client, classifier = client_with(lambda _: pipeline)

    with client:
        inference = client.portal.start_task_soon(
            classifier.classify, "study-1", b"scan"
        )
        while not pipeline.threads:
            time.sleep(0.01)
        started = time.perf_counter()
        response = client.get("/health")
        elapsed = time.perf_counter() - started

        assert response.status_code == 200
        assert not inference.done()
        assert elapsed < 0.5
        assert inference.result().study_id == "study-1"
