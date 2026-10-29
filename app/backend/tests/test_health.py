from fastapi.testclient import TestClient

from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from main import create_app


class UnreachableRepository(InMemoryStudyRepository):
    async def ping(self) -> bool:
        return False


def test_health_returns_ok_with_statuses():
    client = TestClient(create_app(repository=InMemoryStudyRepository()))
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["database_reachable"] is True


def test_health_succeeds_when_database_unreachable():
    client = TestClient(create_app(repository=UnreachableRepository()))
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["database_reachable"] is False
