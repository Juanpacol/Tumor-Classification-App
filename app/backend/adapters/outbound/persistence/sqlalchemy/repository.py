import uuid
from typing import cast

from sqlalchemy import Table, select, text
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from adapters.outbound.persistence.sqlalchemy import mappers
from adapters.outbound.persistence.sqlalchemy.orm import (
    Base,
    ChatMessageRow,
    PredictionRow,
    ReportRow,
    StudyRow,
)
from domain.errors import DomainError
from domain.models import ChatMessage, Prediction, Report, Study

FOREIGN_KEY_VIOLATION = "23503"


def _parse_id(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


def _row_values(row: Base) -> dict[str, object]:
    table = cast(Table, row.__table__)
    return {column.name: getattr(row, column.name) for column in table.columns}


def _insert(row: Base) -> Insert:
    return insert(cast(Table, row.__table__)).values(_row_values(row))


def _upsert(row: Base) -> Insert:
    keys = [column.name for column in cast(Table, row.__table__).primary_key.columns]
    updates = {name: value for name, value in _row_values(row).items() if name not in keys}
    return _insert(row).on_conflict_do_update(index_elements=keys, set_=updates)


class SqlAlchemyStudyRepository:
    """PostgreSQL implementation of StudyRepositoryPort for the whole study aggregate."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get(self, study_id: str) -> Study | None:
        key = _parse_id(study_id)
        if key is None:
            return None
        async with self._session_factory() as session:
            row = await session.get(StudyRow, key)
            return mappers.study_to_domain(row) if row else None

    async def save(self, study: Study) -> None:
        await self._execute(_upsert(mappers.study_to_row(study)))

    async def ping(self) -> bool:
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except (SQLAlchemyError, OSError):
            return False

    async def save_prediction(self, prediction: Prediction) -> None:
        await self._execute(_upsert(mappers.prediction_to_row(prediction)))

    async def get_prediction(self, study_id: str) -> Prediction | None:
        key = _parse_id(study_id)
        if key is None:
            return None
        async with self._session_factory() as session:
            row = await session.get(PredictionRow, key)
            return mappers.prediction_to_domain(row) if row else None

    async def save_report(self, report: Report) -> None:
        await self._execute(_upsert(mappers.report_to_row(report)))

    async def get_report(self, study_id: str) -> Report | None:
        key = _parse_id(study_id)
        if key is None:
            return None
        async with self._session_factory() as session:
            row = await session.get(ReportRow, key)
            return mappers.report_to_domain(row) if row else None

    async def add_message(self, message: ChatMessage) -> None:
        await self._execute(_insert(mappers.message_to_row(message)))

    async def list_messages(self, study_id: str) -> list[ChatMessage]:
        key = _parse_id(study_id)
        if key is None:
            return []
        statement = (
            select(ChatMessageRow)
            .where(ChatMessageRow.study_id == key)
            .order_by(ChatMessageRow.created_at, ChatMessageRow.id)
        )
        async with self._session_factory() as session:
            rows = (await session.scalars(statement)).all()
            return [mappers.message_to_domain(row) for row in rows]

    async def _execute(self, statement: Insert) -> None:
        try:
            async with self._session_factory.begin() as session:
                await session.execute(statement)
        except IntegrityError as error:
            if getattr(error.orig, "sqlstate", None) == FOREIGN_KEY_VIOLATION:
                raise DomainError("STUDY_NOT_FOUND", "Study does not exist") from error
            raise
