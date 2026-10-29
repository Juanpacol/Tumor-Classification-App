from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class StudyStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class ReportStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Study:
    id: str
    file_key: str
    original_format: str
    status: StudyStatus
    created_at: datetime
    clinical_notes: str | None = None
    rejection_reason: str | None = None


@dataclass
class Prediction:
    study_id: str
    label: str
    probabilities: dict[str, float]
    confidence: float
    needs_expert_triage: bool
    quality: dict[str, Any]
    top_features: list[dict[str, Any]]
    model_version: str


@dataclass
class ClinicalReport:
    study_id: str
    findings: str
    visual_observations: str
    impression: str
    differential_considerations: str
    suggested_next_steps: list[str]
    ai_agreement: str


@dataclass
class Report:
    study_id: str
    status: ReportStatus
    model: str
    created_at: datetime
    content: ClinicalReport | None = None
    error: str | None = None


@dataclass
class ChatMessage:
    id: str
    study_id: str
    role: str
    content: str
    created_at: datetime
