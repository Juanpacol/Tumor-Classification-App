import pytest

from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from domain.errors import DomainError
from domain.models import ReportStatus, StudyStatus
from tests.factories import make_message, make_prediction, make_report, make_study


async def test_study_round_trip_and_unknown_study():
    repo = InMemoryStudyRepository()
    study = make_study()

    await repo.save(study)

    assert await repo.get(study.id) == study
    assert await repo.get("missing") is None


async def test_saving_study_again_updates_it():
    repo = InMemoryStudyRepository()
    study = make_study()
    await repo.save(study)

    study.status = StudyStatus.REJECTED
    study.rejection_reason = "Not a brain MRI"
    await repo.save(study)

    stored = await repo.get(study.id)
    assert stored is not None
    assert stored.status == StudyStatus.REJECTED
    assert stored.rejection_reason == "Not a brain MRI"


async def test_prediction_is_replaced_and_missing_returns_none():
    repo = InMemoryStudyRepository()
    study = make_study()
    await repo.save(study)

    await repo.save_prediction(make_prediction(study.id))
    latest = make_prediction(study.id, label="no_tumor")
    await repo.save_prediction(latest)

    assert await repo.get_prediction(study.id) == latest
    assert await repo.get_prediction("missing") is None


async def test_report_progresses_from_pending_to_completed():
    repo = InMemoryStudyRepository()
    study = make_study()
    await repo.save(study)

    await repo.save_report(make_report(study.id, status=ReportStatus.PENDING, content=None))
    completed = make_report(study.id)
    await repo.save_report(completed)

    assert await repo.get_report(study.id) == completed
    assert await repo.get_report("missing") is None


async def test_messages_listed_oldest_first_and_empty_by_default():
    repo = InMemoryStudyRepository()
    study = make_study()
    await repo.save(study)

    later, earlier, middle = (make_message(study.id, m) for m in (10, 0, 5))
    for message in (later, earlier, middle):
        await repo.add_message(message)

    assert await repo.list_messages(study.id) == [earlier, middle, later]
    assert await repo.list_messages("other") == []


async def test_children_of_unknown_study_are_rejected():
    repo = InMemoryStudyRepository()
    study_id = make_study().id

    with pytest.raises(DomainError):
        await repo.save_prediction(make_prediction(study_id))
    with pytest.raises(DomainError):
        await repo.save_report(make_report(study_id))
    with pytest.raises(DomainError):
        await repo.add_message(make_message(study_id))
    assert await repo.get_prediction(study_id) is None


async def test_ping_is_always_true():
    assert await InMemoryStudyRepository().ping() is True
