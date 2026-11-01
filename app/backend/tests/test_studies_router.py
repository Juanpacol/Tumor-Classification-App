import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from src.pipelines import FakeInferencePipeline

from adapters.inbound.http.router_studies import MAX_PAGE_SIZE
from adapters.outbound.llm.fake_llm import FakeLLM
from adapters.outbound.ml.ml_engine_classifier import MLEngineClassifier
from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from adapters.outbound.storage.in_memory_storage import InMemoryStorage
from domain.models import StudyStatus
from main import create_app
from tests.doubles import ScriptedClassifier, ScriptedLLM
from tests.factories import (
    BASE_TIME,
    image_bytes,
    make_prediction,
    make_report,
    make_study,
)

PNG = image_bytes("PNG")


def client_with(repository=None, storage=None, classifier=None, llm=None) -> TestClient:
    app = create_app(
        llm=llm or FakeLLM(),
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
            files={"file": ("scan.png", PNG, "image/png")},
            data={"clinical_notes": "Headache"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["clinical_notes"] == "Headache"
    assert body["prediction"]["label"] == "no_tumor"


def test_upload_of_a_file_that_is_not_an_image_is_415():
    with client_with() as client:
        response = client.post(
            "/api/studies", files={"file": ("scan.dcm", b"DICM", "application/dicom")}
        )

    assert response.status_code == 415
    assert response.json()["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_upload_over_20_mb_is_413():
    with client_with() as client:
        response = client.post(
            "/api/studies",
            files={"file": ("scan.png", PNG + b"0" * (20 * 1024 * 1024), "image/png")},
        )

    assert response.status_code == 413


def test_upload_with_notes_over_2000_characters_is_422():
    with client_with() as client:
        response = client.post(
            "/api/studies",
            files={"file": ("scan.png", PNG, "image/png")},
            data={"clinical_notes": "x" * 2001},
        )

    assert response.status_code == 422


def test_get_unknown_study_is_404():
    with client_with() as client:
        response = client.get(f"/api/studies/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["code"] == "STUDY_NOT_FOUND"


@pytest.mark.parametrize("path", ["/api/studies/missing", "/api/studies/missing/image"])
def test_a_study_id_that_is_not_a_uuid_is_422(path):
    with client_with() as client:
        response = client.get(path)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "query", ["page=0", "page_size=0", f"page_size={MAX_PAGE_SIZE + 1}"]
)
def test_list_studies_rejects_a_page_out_of_range(query):
    with client_with() as client:
        response = client.get(f"/api/studies?{query}")

    assert response.status_code == 422


async def test_list_studies_pages_newest_first():
    repository = InMemoryStudyRepository()
    studies = [
        make_study(created_at=BASE_TIME + timedelta(minutes=minute))
        for minute in range(3)
    ]
    for study in studies:
        await repository.save(study)

    with client_with(repository=repository) as client:
        response = client.get("/api/studies?page=2&page_size=2")

    body = response.json()
    assert [item["id"] for item in body["items"]] == [studies[0].id]
    assert (body["page"], body["page_size"], body["total"]) == (2, 2, 3)


async def test_get_study_exposes_the_report_citations():
    repository = InMemoryStudyRepository()
    study = make_study()
    await repository.save(study)
    await repository.save_report(make_report(study.id))

    with client_with(repository=repository) as client:
        response = client.get(f"/api/studies/{study.id}")

    [citation] = response.json()["report"]["content"]["citations"]
    assert citation == {
        "claim": "Mass in left frontal lobe",
        "evidence": [{"key": "prediction.label", "description": "glioma"}],
        "verified": True,
    }


async def completed_study_with_image(repository, storage):
    study = make_study(status=StudyStatus.COMPLETED)
    await repository.save(study)
    await repository.save_prediction(make_prediction(study.id))
    storage.save(b"image", study.file_key)
    return study


async def test_post_report_returns_the_report_with_verified_citations():
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    study = await completed_study_with_image(repository, storage)

    with client_with(repository=repository, storage=storage) as client:
        response = client.post(f"/api/studies/{study.id}/report")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    citations = body["content"]["citations"]
    assert all(c["verified"] == bool(c["evidence"]) for c in citations)
    assert any(c["verified"] for c in citations)
    assert (await repository.get_report(study.id)).status.value == "completed"


def test_post_report_of_an_unknown_study_is_404():
    with client_with() as client:
        response = client.post("/api/studies/missing/report")

    assert response.status_code == 404


async def test_post_report_without_a_prediction_is_409():
    repository = InMemoryStudyRepository()
    study = make_study(status=StudyStatus.COMPLETED)
    await repository.save(study)

    with client_with(repository=repository) as client:
        response = client.post(f"/api/studies/{study.id}/report")

    assert response.status_code == 409
    assert response.json()["code"] == "PREDICTION_NOT_FOUND"


async def test_post_report_with_an_invalid_model_output_is_502_and_stored_as_failed():
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    study = await completed_study_with_image(repository, storage)
    llm = ScriptedLLM({}, {})

    with client_with(repository=repository, storage=storage, llm=llm) as client:
        response = client.post(f"/api/studies/{study.id}/report")

    assert response.status_code == 502
    assert (await repository.get_report(study.id)).status.value == "failed"


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
            files={"file": ("scan.png", PNG, "image/png")},
        )
        study_id = upload.json()["id"]

        response = client.get(f"/api/studies/{study_id}/image")

    assert response.status_code == 200
    assert response.content == PNG
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
