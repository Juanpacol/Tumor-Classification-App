from fastapi import FastAPI
from fastapi.testclient import TestClient

from adapters.inbound.http.error_handlers import register_exception_handlers
from domain.errors import DomainError


def test_domain_error_maps_to_consistent_json_body():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        raise DomainError(code="STUDY_NOT_FOUND", message="Study does not exist")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom")

    assert response.status_code == 400
    assert response.json() == {
        "code": "STUDY_NOT_FOUND",
        "message": "Study does not exist",
    }
