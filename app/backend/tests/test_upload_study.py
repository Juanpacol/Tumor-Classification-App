from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from adapters.outbound.storage.in_memory_storage import InMemoryStorage
from application.upload_study import UploadStudy
from domain.errors import QualityCheckFailedError, UnsupportedImageError
from domain.models import StudyStatus
from tests.doubles import ScriptedClassifier
from tests.factories import make_prediction


async def test_a_classified_image_becomes_a_completed_study():
    repo = InMemoryStudyRepository()
    storage = InMemoryStorage()
    prediction = make_prediction("placeholder")
    use_case = UploadStudy(repo, storage, ScriptedClassifier(prediction))

    detail = await use_case.execute(image=b"scan", original_format="png")

    assert detail.study.status is StudyStatus.COMPLETED
    assert detail.prediction.label == prediction.label
    assert storage.load(detail.study.file_key) == b"scan"
    assert (await repo.get(detail.study.id)).status is StudyStatus.COMPLETED


async def test_an_unsupported_image_becomes_a_rejected_study():
    use_case = UploadStudy(
        InMemoryStudyRepository(),
        InMemoryStorage(),
        ScriptedClassifier(UnsupportedImageError("not an MRI")),
    )

    detail = await use_case.execute(image=b"not-an-image", original_format="png")

    assert detail.study.status is StudyStatus.REJECTED
    assert detail.prediction is None


async def test_a_failed_quality_check_becomes_a_rejected_study():
    use_case = UploadStudy(
        InMemoryStudyRepository(),
        InMemoryStorage(),
        ScriptedClassifier(QualityCheckFailedError("too noisy", {"snr": 1.0})),
    )

    detail = await use_case.execute(image=b"scan", original_format="png")

    assert detail.study.status is StudyStatus.REJECTED
    assert detail.study.rejection_reason is not None
