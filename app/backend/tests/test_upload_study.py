import pytest

from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from adapters.outbound.storage.in_memory_storage import InMemoryStorage
from application.upload_study import UploadStudy
from domain.errors import (
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


async def test_an_unreadable_image_becomes_a_rejected_study():
    use_case = UploadStudy(
        InMemoryStudyRepository(),
        InMemoryStorage(),
        ScriptedClassifier(UnsupportedImageError("not an MRI")),
    )

    detail = await use_case.execute(image=PNG)

    assert detail.study.status is StudyStatus.REJECTED
    assert detail.prediction is None


async def test_a_failed_quality_check_becomes_a_rejected_study():
    use_case = UploadStudy(
        InMemoryStudyRepository(),
        InMemoryStorage(),
        ScriptedClassifier(QualityCheckFailedError("too noisy", {"snr": 1.0})),
    )

    detail = await use_case.execute(image=PNG)

    assert detail.study.status is StudyStatus.REJECTED
    assert detail.study.rejection_reason is not None


@pytest.mark.parametrize(
    "error", [ModelUnavailableError("no bundle"), RuntimeError("boom")]
)
async def test_a_classifier_failure_is_stored_as_failed_and_raised(error):
    repo = InMemoryStudyRepository()
    use_case = UploadStudy(repo, InMemoryStorage(), ScriptedClassifier(error))

    with pytest.raises(type(error)):
        await use_case.execute(image=PNG)

    [study], _ = await repo.list_recent(1, 20)
    assert study.status is StudyStatus.FAILED
