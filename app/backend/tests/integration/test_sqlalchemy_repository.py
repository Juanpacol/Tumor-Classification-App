import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from adapters.outbound.persistence.sqlalchemy.engine import (
    create_engine,
    create_session_factory,
)
from adapters.outbound.persistence.sqlalchemy.repository import SqlAlchemyStudyRepository
from domain.errors import DomainError
from domain.models import ReportStatus, StudyStatus
from settings import Settings
from tests.factories import make_message, make_prediction, make_report, make_study


async def test_saved_study_readable_from_a_fresh_engine(repository: SqlAlchemyStudyRepository):
    study = make_study()
    await repository.save(study)

    fresh_engine = create_engine(Settings())
    try:
        stored = await SqlAlchemyStudyRepository(create_session_factory(fresh_engine)).get(study.id)
    finally:
        await fresh_engine.dispose()

    assert stored == study


async def test_unknown_or_malformed_study_returns_none(repository: SqlAlchemyStudyRepository):
    assert await repository.get(str(uuid.uuid4())) is None
    assert await repository.get("not-a-uuid") is None


async def test_saving_study_again_updates_without_duplicating(
    repository: SqlAlchemyStudyRepository, engine: AsyncEngine
):
    study = make_study()
    await repository.save(study)
    study.status = StudyStatus.REJECTED
    study.rejection_reason = "Not a brain MRI"

    await repository.save(study)

    assert await repository.get(study.id) == study
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT count(*) FROM studies")) == 1


async def test_invalid_status_rejected_by_database(engine: AsyncEngine):
    with pytest.raises(IntegrityError):
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO studies (id, created_at, status, original_format, file_key) "
                    "VALUES (:id, now(), 'archived', 'dicom', 'k')"
                ),
                {"id": uuid.uuid4()},
            )


async def test_ping_reports_reachable_database(repository: SqlAlchemyStudyRepository):
    assert await repository.ping() is True


async def test_ping_returns_false_when_database_unreachable():
    engine = create_engine(Settings(database_url="postgresql://x:y@localhost:5499/none"))
    try:
        assert await SqlAlchemyStudyRepository(create_session_factory(engine)).ping() is False
    finally:
        await engine.dispose()


async def test_prediction_round_trip_and_replacement(
    repository: SqlAlchemyStudyRepository, engine: AsyncEngine
):
    study = make_study()
    await repository.save(study)
    await repository.save_prediction(make_prediction(study.id))
    latest = make_prediction(study.id, label="no_tumor", top_features=[], quality={"snr": 3.0})

    await repository.save_prediction(latest)

    assert await repository.get_prediction(study.id) == latest
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT count(*) FROM predictions")) == 1


async def test_missing_prediction_and_report_return_none(repository: SqlAlchemyStudyRepository):
    study = make_study()
    await repository.save(study)

    assert await repository.get_prediction(study.id) is None
    assert await repository.get_report(study.id) is None


async def test_report_progresses_from_pending_to_completed(
    repository: SqlAlchemyStudyRepository,
):
    study = make_study()
    await repository.save(study)
    await repository.save_report(make_report(study.id, status=ReportStatus.PENDING, content=None))
    completed = make_report(study.id)

    await repository.save_report(completed)

    assert await repository.get_report(study.id) == completed


async def test_failed_report_keeps_error(repository: SqlAlchemyStudyRepository):
    study = make_study()
    await repository.save(study)
    failed = make_report(study.id, status=ReportStatus.FAILED, content=None, error="LLM timeout")

    await repository.save_report(failed)

    assert await repository.get_report(study.id) == failed


async def test_children_of_unknown_study_are_rejected(
    repository: SqlAlchemyStudyRepository, engine: AsyncEngine
):
    study_id = make_study().id

    with pytest.raises(DomainError) as error:
        await repository.save_prediction(make_prediction(study_id))
    with pytest.raises(DomainError):
        await repository.save_report(make_report(study_id))
    with pytest.raises(DomainError):
        await repository.add_message(make_message(study_id))

    assert error.value.code == "STUDY_NOT_FOUND"
    async with engine.connect() as connection:
        counts = await connection.execute(
            text(
                "SELECT (SELECT count(*) FROM predictions), (SELECT count(*) FROM reports), "
                "(SELECT count(*) FROM chat_messages)"
            )
        )
        assert counts.one() == (0, 0, 0)


async def test_messages_listed_oldest_first(repository: SqlAlchemyStudyRepository):
    study = make_study()
    await repository.save(study)
    later, earlier, middle = (make_message(study.id, m) for m in (10, 0, 5))
    for message in (later, earlier, middle):
        await repository.add_message(message)

    assert await repository.list_messages(study.id) == [earlier, middle, later]


async def test_study_without_messages_lists_empty(repository: SqlAlchemyStudyRepository):
    study = make_study()
    await repository.save(study)

    assert await repository.list_messages(study.id) == []
    assert await repository.list_messages("not-a-uuid") == []


async def test_deleting_study_cascades_to_children(
    repository: SqlAlchemyStudyRepository, engine: AsyncEngine
):
    study = make_study()
    await repository.save(study)
    await repository.save_prediction(make_prediction(study.id))
    await repository.save_report(make_report(study.id))
    await repository.add_message(make_message(study.id))

    async with engine.begin() as connection:
        await connection.execute(text("DELETE FROM studies WHERE id = :id"), {"id": study.id})

    assert await repository.get_prediction(study.id) is None
    assert await repository.get_report(study.id) is None
    assert await repository.list_messages(study.id) == []
