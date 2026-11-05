from fastapi import FastAPI
from fastapi.testclient import TestClient

from adapters.inbound.http.security import (
    SecurityHeadersMiddleware,
    UploadRateLimitMiddleware,
)


def _app(limit: int = 2) -> TestClient:
    app = FastAPI()
    app.add_middleware(UploadRateLimitMiddleware, limit=limit)
    app.add_middleware(SecurityHeadersMiddleware)

    @app.post("/api/studies")
    def upload() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/api/studies")
    def listing() -> dict[str, str]:
        return {"ok": "yes"}

    return TestClient(app)


def test_responses_carry_security_headers() -> None:
    response = _app().get("/api/studies")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_uploads_beyond_the_limit_get_429() -> None:
    client = _app(limit=2)
    assert client.post("/api/studies").status_code == 200
    assert client.post("/api/studies").status_code == 200
    blocked = client.post("/api/studies")
    assert blocked.status_code == 429
    assert blocked.json()["code"] == "TOO_MANY_UPLOADS"
    assert blocked.headers["x-content-type-options"] == "nosniff"


def test_reads_are_never_limited() -> None:
    client = _app(limit=1)
    assert all(client.get("/api/studies").status_code == 200 for _ in range(5))


def test_limit_is_per_client_address() -> None:
    client = _app(limit=1)
    assert (
        client.post("/api/studies", headers={"x-forwarded-for": "1.1.1.1"}).status_code
        == 200
    )
    assert (
        client.post("/api/studies", headers={"x-forwarded-for": "2.2.2.2"}).status_code
        == 200
    )
    assert (
        client.post("/api/studies", headers={"x-forwarded-for": "1.1.1.1"}).status_code
        == 429
    )
