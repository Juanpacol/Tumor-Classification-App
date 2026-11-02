import io
import logging
import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from src.pipelines import FakeInferencePipeline

from adapters.inbound.http.router_studies import MAX_PAGE_SIZE
from adapters.outbound.llm.fake_llm import FakeLLM
from adapters.outbound.ml.ml_engine_classifier import MLEngineClassifier
from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from adapters.outbound.storage.in_memory_storage import InMemoryStorage
from domain.errors import QualityCheckFailedError, UnsupportedImageError
from domain.models import StudyStatus
from main import create_app
from tests.doubles import ScriptedClassifier, ScriptedLLM
from tests.factories import (
    BASE_TIME,
    image_bytes,
    make_prediction,
    make_report,
    make_study,
    patterned_image,
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


async def test_upload_study_returns_a_completed_study():
    repository = InMemoryStudyRepository()
    prediction = make_prediction("placeholder", label="no_tumor")
    notes = "x" * 2000
    with client_with(repository, classifier=ScriptedClassifier(prediction)) as client:
        response = client.post(
            "/api/studies",
            files={"file": ("scan.png", PNG, "image/png")},
            data={"clinical_notes": notes},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["clinical_notes"] == notes
    assert body["prediction"]["label"] == "no_tumor"
    assert (await repository.get(body["id"])).status is StudyStatus.COMPLETED
    assert (await repository.get_prediction(body["id"])).label == "no_tumor"


@pytest.mark.parametrize(
    "error, code",
    [
        (QualityCheckFailedError("too noisy", {"snr": 1.0}), "QUALITY_CHECK_FAILED"),
        (UnsupportedImageError("image is 32x32"), "UNSUPPORTED_IMAGE"),
    ],
    ids=["quality-check", "unsupported-image"],
)
async def test_a_rejected_upload_is_422_with_the_stored_study(error, code):
    repository = InMemoryStudyRepository()
    with client_with(repository, classifier=ScriptedClassifier(error)) as client:
        response = client.post(
            "/api/studies", files={"file": ("scan.png", PNG, "image/png")}
        )

    assert response.status_code == 422
    body = response.json()
    reason = error.details["reason"]
    assert body["code"] == code
    assert body["details"]["reason"] == reason
    assert body["study"]["status"] == "rejected"
    assert body["study"]["rejection_reason"] == reason
    assert body["study"]["prediction"] is None
    stored = await repository.get(body["study"]["id"])
    assert stored.status is StudyStatus.REJECTED
    assert stored.rejection_reason == reason


async def test_an_unexpected_classifier_error_is_a_generic_500_without_phi(caplog):
    repository = InMemoryStudyRepository()
    classifier = ScriptedClassifier(RuntimeError("boom at /data/jane_roe"))
    with (
        caplog.at_level(logging.DEBUG),
        client_with(repository, classifier=classifier) as client,
    ):
        response = client.post(
            "/api/studies",
            files={"file": ("jane_roe.png", PNG, "image/png")},
            data={"clinical_notes": "Patient Jane Roe, 54"},
        )

    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert "boom" not in response.text and "jane_roe" not in response.text
    [study], _ = await repository.list_recent(1, 20)
    assert study.status is StudyStatus.FAILED
    assert study.id in caplog.text and "RuntimeError" in caplog.text
    for phi in ("Jane Roe", "jane_roe", "boom"):
        assert phi not in caplog.text


TEXT_AS_PNG = ("scan.png", b"not an image at all", "image/png")
DICOM = ("scan.dcm", b"\x00" * 128 + b"DICM" + b"\x00" * 64, "application/dicom")
TRUNCATED_JPEG = ("scan.jpg", patterned_image("JPEG")[:-200], "image/jpeg")
OVERSIZED = ("scan.png", PNG + b"0" * (20 * 1024 * 1024 + 1 - len(PNG)), "image/png")


@pytest.mark.parametrize(
    "file, notes, ready, status, code",
    [
        (("scan.png", PNG, "image/png"), "x" * 2001, True, 422, None),
        (None, None, True, 422, None),
        (OVERSIZED, None, True, 413, "FILE_TOO_LARGE"),
        (DICOM, None, True, 415, "UNSUPPORTED_MEDIA_TYPE"),
        (TEXT_AS_PNG, None, True, 415, "UNSUPPORTED_MEDIA_TYPE"),
        (TRUNCATED_JPEG, None, True, 415, "UNSUPPORTED_MEDIA_TYPE"),
        (("scan.png", PNG, "image/png"), None, False, 503, "MODEL_UNAVAILABLE"),
    ],
    ids=[
        "notes-too-long",
        "file-missing",
        "over-20-mb",
        "dicom",
        "text-renamed-png",
        "truncated-jpeg",
        "no-model",
    ],
)
async def test_a_refused_upload_stores_nothing(file, notes, ready, status, code):
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    classifier = ScriptedClassifier(make_prediction("placeholder"), ready=ready)
    with client_with(repository, storage, classifier) as client:
        response = client.post(
            "/api/studies",
            files={"file": file} if file else {"other": ("x.txt", b"x", "text/plain")},
            data={"clinical_notes": notes} if notes else {},
        )

    assert response.status_code == status
    if code:
        assert response.json()["code"] == code
    assert (await repository.list_recent(1, 20))[1] == 0
    assert storage._store == {}


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


@pytest.mark.parametrize("fmt, mime", [("JPEG", "image/jpeg"), ("PNG", "image/png")])
def test_the_stored_image_has_no_metadata_and_the_same_pixels(fmt, mime):
    storage = InMemoryStorage()
    original = image_bytes(fmt, with_metadata=True)
    classifier = ScriptedClassifier(make_prediction("placeholder"))
    with client_with(storage=storage, classifier=classifier) as client:
        response = client.post("/api/studies", files={"file": ("scan", original, mime)})

    study_id = response.json()["id"]
    assert list(storage._store) == [f"studies/{uuid.UUID(study_id)}"]
    stored = storage.load(f"studies/{study_id}")
    with (
        Image.open(io.BytesIO(stored)) as clean,
        Image.open(io.BytesIO(original)) as source,
    ):
        assert source.getexif()
        assert not clean.getexif()
        assert not getattr(clean, "text", {})
        assert clean.tobytes() == source.tobytes()
    for chunk in (b"eXIf", b"tEXt", b"iTXt", b"Jane Doe", b"Scanner X200"):
        assert chunk not in stored


async def test_the_original_filename_is_not_kept():
    repository, storage = InMemoryStudyRepository(), InMemoryStorage()
    classifier = ScriptedClassifier(make_prediction("placeholder"))
    with client_with(repository, storage, classifier) as client:
        response = client.post(
            "/api/studies", files={"file": ("john_doe_mri.png", PNG, "image/png")}
        )

    study = await repository.get(response.json()["id"])
    assert "john_doe_mri" not in response.text
    assert "john_doe_mri" not in repr(study)
    assert all("john_doe_mri" not in key for key in storage._store)


def test_health_classifier_default_still_works():
    """create_app's default MLEngineClassifier isn't broken by the new storage arg."""
    app = create_app(
        repository=InMemoryStudyRepository(),
        classifier=MLEngineClassifier(loader=lambda _: FakeInferencePipeline()),
    )
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
