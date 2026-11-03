import pytest

from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from adapters.outbound.storage.in_memory_storage import InMemoryStorage
from application.upload_study import StudyRejectedError, UploadStudy
from domain.errors import (
    ClassificationFailedError,
    ModelUnavailableError,
    QualityCheckFailedError,
    UnsupportedImageError,
    UnsupportedMediaTypeError,
)
from domain.models import StudyStatus
from tests.doubles import ScriptedClassifier
from tests.factories import image_bytes, make_prediction

PNG = image_bytes("PNG")


async def test_a_classified_image_becomes_a_completed_study():
    repo = InMemoryStudyRepository()
    storage = InMemoryStorage()
    prediction = make_prediction("placeholder")
    use_case = UploadStudy(repo, storage, ScriptedClassifier(prediction))

    detail = await use_case.execute(image=PNG)

    assert detail.study.status is StudyStatus.COMPLETED
    assert detail.study.original_format == "png"
    assert detail.prediction.label == prediction.label
    assert storage.load(detail.study.file_key) == PNG
    assert (await repo.get(detail.study.id)).status is StudyStatus.COMPLETED


async def test_the_stored_and_classified_image_has_no_metadata():
    storage = InMemoryStorage()
    use_case = UploadStudy(
        InMemoryStudyRepository(),
        storage,
        ScriptedClassifier(make_prediction("placeholder")),
    )

    detail = await use_case.execute(image=image_bytes("JPEG", with_metadata=True))

    assert b"Jane Doe" not in storage.load(detail.study.file_key)


async def test_an_unsupported_file_is_refused_and_nothing_is_stored():
    repo, storage = InMemoryStudyRepository(), InMemoryStorage()
    use_case = UploadStudy(repo, storage, ScriptedClassifier())

    with pytest.raises(UnsupportedMediaTypeError):
        await use_case.execute(image=b"DICM-not-an-image")

    assert (await repo.list_recent(1, 20))[1] == 0
    assert storage._store == {}


async def test_an_upload_without_a_model_is_refused_and_nothing_is_stored():
    repo, storage = InMemoryStudyRepository(), InMemoryStorage()
    use_case = UploadStudy(repo, storage, ScriptedClassifier(ready=False))

    with pytest.raises(ModelUnavailableError) as error:
        await use_case.execute(image=PNG)

    assert error.value.status_code == 503
    assert (await repo.list_recent(1, 20))[1] == 0
    assert storage._store == {}


@pytest.mark.parametrize(
    "error, code",
    [
        (UnsupportedImageError("too small"), "UNSUPPORTED_IMAGE"),
        (QualityCheckFailedError("too noisy", {"snr": 1.0}), "QUALITY_CHECK_FAILED"),
    ],
    ids=["unsupported-image", "quality-check"],
)
async def test_a_rejected_image_is_stored_as_rejected_with_its_reason(error, code):
    repo = InMemoryStudyRepository()
    use_case = UploadStudy(repo, InMemoryStorage(), ScriptedClassifier(error))

    with pytest.raises(StudyRejectedError) as raised:
        await use_case.execute(image=PNG)

    reason = error.details["reason"]
    assert raised.value.code == code
    assert raised.value.status_code == 422
    assert raised.value.details["reason"] == reason
    assert raised.value.detail.prediction is None
    stored = await repo.get(raised.value.detail.study.id)
    assert stored.status is StudyStatus.REJECTED
    assert stored.rejection_reason == reason
    assert await repo.get_prediction(stored.id) is None


async def test_a_domain_error_from_the_classifier_is_stored_as_failed_and_raised():
    repo = InMemoryStudyRepository()
    error = ModelUnavailableError("unloaded")
    use_case = UploadStudy(repo, InMemoryStorage(), ScriptedClassifier(error))

    with pytest.raises(ModelUnavailableError):
        await use_case.execute(image=PNG)

    [study], _ = await repo.list_recent(1, 20)
    assert study.status is StudyStatus.FAILED


async def test_an_unexpected_error_is_stored_as_failed_and_raised_as_generic():
    repo = InMemoryStudyRepository()
    use_case = UploadStudy(
        repo, InMemoryStorage(), ScriptedClassifier(RuntimeError("boom"))
    )

    with pytest.raises(ClassificationFailedError) as raised:
        await use_case.execute(image=PNG)

    assert raised.value.status_code == 500
    assert "boom" not in raised.value.message
    assert raised.value.__cause__ is None
    [study], _ = await repo.list_recent(1, 20)
    assert study.status is StudyStatus.FAILED
