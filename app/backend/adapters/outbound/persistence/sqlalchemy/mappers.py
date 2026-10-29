import uuid
from dataclasses import asdict

from adapters.outbound.persistence.sqlalchemy.orm import (
    ChatMessageRow,
    PredictionRow,
    ReportRow,
    StudyRow,
)
from domain.models import (
    ChatMessage,
    ClinicalReport,
    Prediction,
    Report,
    ReportStatus,
    Study,
    StudyStatus,
)


def study_to_row(study: Study) -> StudyRow:
    return StudyRow(
        id=uuid.UUID(study.id),
        created_at=study.created_at,
        status=study.status.value,
        original_format=study.original_format,
        file_key=study.file_key,
        clinical_notes=study.clinical_notes,
        rejection_reason=study.rejection_reason,
    )


def study_to_domain(row: StudyRow) -> Study:
    return Study(
        id=str(row.id),
        file_key=row.file_key,
        original_format=row.original_format,
        status=StudyStatus(row.status),
        created_at=row.created_at,
        clinical_notes=row.clinical_notes,
        rejection_reason=row.rejection_reason,
    )


def prediction_to_row(prediction: Prediction) -> PredictionRow:
    return PredictionRow(
        study_id=uuid.UUID(prediction.study_id),
        label=prediction.label,
        probabilities=prediction.probabilities,
        confidence=prediction.confidence,
        needs_expert_triage=prediction.needs_expert_triage,
        quality=prediction.quality,
        top_features=prediction.top_features,
        model_version=prediction.model_version,
    )


def prediction_to_domain(row: PredictionRow) -> Prediction:
    return Prediction(
        study_id=str(row.study_id),
        label=row.label,
        probabilities=row.probabilities,
        confidence=row.confidence,
        needs_expert_triage=row.needs_expert_triage,
        quality=row.quality,
        top_features=row.top_features,
        model_version=row.model_version,
    )


def report_to_row(report: Report) -> ReportRow:
    return ReportRow(
        study_id=uuid.UUID(report.study_id),
        status=report.status.value,
        content=asdict(report.content) if report.content is not None else None,
        model=report.model,
        error=report.error,
        created_at=report.created_at,
    )


def report_to_domain(row: ReportRow) -> Report:
    return Report(
        study_id=str(row.study_id),
        status=ReportStatus(row.status),
        model=row.model,
        created_at=row.created_at,
        content=ClinicalReport(**row.content) if row.content is not None else None,
        error=row.error,
    )


def message_to_row(message: ChatMessage) -> ChatMessageRow:
    return ChatMessageRow(
        id=uuid.UUID(message.id),
        study_id=uuid.UUID(message.study_id),
        role=message.role,
        content=message.content,
        created_at=message.created_at,
    )


def message_to_domain(row: ChatMessageRow) -> ChatMessage:
    return ChatMessage(
        id=str(row.id),
        study_id=str(row.study_id),
        role=row.role,
        content=row.content,
        created_at=row.created_at,
    )
