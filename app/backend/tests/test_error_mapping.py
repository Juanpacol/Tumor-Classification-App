import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from adapters.inbound.http.error_handlers import register_exception_handlers
from domain.errors import (
    DomainError,
    ModelUnavailableError,
    QualityCheckFailedError,
    UnsupportedImageError,
)

METRICS = {"snr": 1.5, "entropy": 0.8, "skull_stripped": False}


def _client_raising(error: DomainError) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        raise error

    return TestClient(app, raise_server_exceptions=False)


def test_domain_error_maps_to_consistent_json_body():
    error = DomainError(code="STUDY_NOT_FOUND", message="Study does not exist")
    response = _client_raising(error).get("/boom")

    assert response.status_code == 400
    assert response.json() == {
        "code": "STUDY_NOT_FOUND",
        "message": "Study does not exist",
    }


@pytest.mark.parametrize(
    ("error", "code", "status_code", "details"),
    [
        (ModelUnavailableError("no bundle"), "MODEL_UNAVAILABLE", 503, None),
        (
            UnsupportedImageError("empty input"),
            "UNSUPPORTED_IMAGE",
            422,
            {"reason": "empty input"},
        ),
        (
            QualityCheckFailedError("snr below threshold", METRICS),
            "QUALITY_CHECK_FAILED",
            422,
            {"reason": "snr below threshold", **METRICS},
        ),
    ],
)
def test_classification_errors_carry_code_status_and_details(
    error: DomainError, code: str, status_code: int, details: dict | None
):
    assert (error.code, error.status_code, error.details) == (
        code,
        status_code,
        details,
    )


def test_model_unavailable_body_has_no_details():
    response = _client_raising(ModelUnavailableError("no bundle")).get("/boom")

    assert response.status_code == 503
    assert set(response.json()) == {"code", "message"}
    assert response.json()["code"] == "MODEL_UNAVAILABLE"


def test_quality_error_body_keeps_reason_and_metrics():
    error = QualityCheckFailedError("snr below threshold", METRICS)
    response = _client_raising(error).get("/boom")

    assert response.status_code == 422
    assert response.json()["code"] == "QUALITY_CHECK_FAILED"
    assert response.json()["details"] == {"reason": "snr below threshold", **METRICS}
