from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok_with_statuses():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["database_reachable"] is True
