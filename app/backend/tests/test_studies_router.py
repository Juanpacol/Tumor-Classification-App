from fastapi.testclient import TestClient
from src.pipelines import FakeInferencePipeline

from adapters.outbound.ml.ml_engine_classifier import MLEngineClassifier
from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from adapters.outbound.storage.in_memory_storage import InMemoryStorage
from main import create_app
from tests.doubles import ScriptedClassifier
from tests.factories import make_prediction


def client_with(repository=None, storage=None, classifier=None) -> TestClient:
    app = create_app(
        repository=repository or InMemoryStudyRepository(),
        storage=storage or InMemoryStorage(),
        classifier=classifier or ScriptedClassifier(),
    )
    return TestClient(app)


def test_list_studies_is_empty_by_default():
    with client_with() as client:
        response = client.get("/api/studies")

    assert response.status_code == 200
    assert response.json() == {"items": [], "page": 1, "page_size": 20, "total": 0}


def test_upload_study_returns_a_completed_study():
    prediction = make_prediction("placeholder", label="no_tumor")
    with client_with(classifier=ScriptedClassifier(prediction)) as client:
        response = client.post(
            "/api/studies",
            files={"file": ("scan.png", b"fake-bytes", "image/png")},
            data={"clinical_notes": "Headache"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["clinical_notes"] == "Headache"
    assert body["prediction"]["label"] == "no_tumor"


def test_get_unknown_study_is_404():
    with client_with() as client:
        response = client.get("/api/studies/missing")

    assert response.status_code == 404
    assert response.json()["code"] == "STUDY_NOT_FOUND"


def test_get_study_image_returns_the_uploaded_bytes():
    repository = InMemoryStudyRepository()
    storage = InMemoryStorage()
    prediction = make_prediction("placeholder")
    with client_with(
        repository=repository,
        storage=storage,
        classifier=ScriptedClassifier(prediction),
    ) as client:
        upload = client.post(
            "/api/studies",
            files={"file": ("scan.png", b"fake-bytes", "image/png")},
        )
        study_id = upload.json()["id"]

        response = client.get(f"/api/studies/{study_id}/image")

    assert response.status_code == 200
    assert response.content == b"fake-bytes"
    assert response.headers["content-type"] == "image/png"


def test_health_classifier_default_still_works():
    """create_app's default MLEngineClassifier isn't broken by the new storage arg."""
    app = create_app(
        repository=InMemoryStudyRepository(),
        classifier=MLEngineClassifier(loader=lambda _: FakeInferencePipeline()),
    )
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
