from fastapi.testclient import TestClient

from main import create_app
from settings import Settings


def test_health_reports_postgres_reachable(migrated_database: Settings):
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["database_reachable"] is True
