import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    MetaData,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from domain.models import ReportStatus, StudyStatus

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def _in_values(column: str, enum: type[StudyStatus] | type[ReportStatus]) -> str:
    values = ", ".join(f"'{member.value}'" for member in enum)
    return f"{column} IN ({values})"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class StudyRow(Base):
    __tablename__ = "studies"
    __table_args__ = (
        CheckConstraint(_in_values("status", StudyStatus), name="status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20))
    original_format: Mapped[str] = mapped_column(String(32))
    file_key: Mapped[str] = mapped_column(String(512))
    clinical_notes: Mapped[str | None] = mapped_column(Text)
    rejection_reason: Mapped[str | None] = mapped_column(Text)


class PredictionRow(Base):
    __tablename__ = "predictions"

    study_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("studies.id", ondelete="CASCADE"), primary_key=True
    )
    label: Mapped[str] = mapped_column(String(64))
    probabilities: Mapped[dict[str, float]] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float)
    needs_expert_triage: Mapped[bool] = mapped_column(Boolean)
    quality: Mapped[dict[str, Any]] = mapped_column(JSONB)
    top_features: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    model_version: Mapped[str] = mapped_column(String(64))


class ReportRow(Base):
    __tablename__ = "reports"
    __table_args__ = (
        CheckConstraint(_in_values("status", ReportStatus), name="status"),
    )

    study_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("studies.id", ondelete="CASCADE"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(20))
    content: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    model: Mapped[str] = mapped_column(String(128))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ChatMessageRow(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_study_id_created_at", "study_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    study_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("studies.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
