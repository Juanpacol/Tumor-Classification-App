from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class StudyStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class Study:
    id: str
    image_path: str
    status: StudyStatus
    created_at: datetime


@dataclass
class Prediction:
    study_id: str
    label: str
    probabilities: dict[str, float]
    confidence: float
    needs_expert_triage: bool
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
class ChatMessage:
    id: str
    study_id: str
    role: str
    content: str
    created_at: datetime
