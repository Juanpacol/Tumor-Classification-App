import uuid
from datetime import UTC, datetime, timedelta

from domain.models import (
    ChatMessage,
    ClinicalReport,
    Prediction,
    Report,
    ReportStatus,
    Study,
    StudyStatus,
)

BASE_TIME = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_study(**overrides: object) -> Study:
    fields: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "file_key": "studies/abc.dcm",
        "original_format": "dicom",
        "status": StudyStatus.PENDING,
        "created_at": BASE_TIME,
        "clinical_notes": "Headache for 3 weeks",
        "rejection_reason": None,
    }
    fields.update(overrides)
    return Study(**fields)  # type: ignore[arg-type]


def make_prediction(study_id: str, **overrides: object) -> Prediction:
    fields: dict[str, object] = {
        "study_id": study_id,
        "label": "glioma",
        "probabilities": {"glioma": 0.7, "meningioma": 0.1, "pituitary": 0.1, "no_tumor": 0.1},
        "confidence": 0.7,
        "needs_expert_triage": True,
        "quality": {"snr": 12.5, "passed": True},
        "top_features": [{"name": "glcm_contrast", "value": 0.42}],
        "model_version": "hybrid-1.0.0",
    }
    fields.update(overrides)
    return Prediction(**fields)  # type: ignore[arg-type]


def make_clinical_report(study_id: str) -> ClinicalReport:
    return ClinicalReport(
        study_id=study_id,
        findings="Mass in left frontal lobe",
        visual_observations="Heatmap focused on lesion",
        impression="Consistent with glioma",
        differential_considerations="Metastasis",
        suggested_next_steps=["Contrast MRI", "Neurosurgery consult"],
        ai_agreement="agree",
    )


def make_report(study_id: str, **overrides: object) -> Report:
    fields: dict[str, object] = {
        "study_id": study_id,
        "status": ReportStatus.COMPLETED,
        "model": "muse-spark-1.3",
        "created_at": BASE_TIME,
        "content": make_clinical_report(study_id),
        "error": None,
    }
    fields.update(overrides)
    return Report(**fields)  # type: ignore[arg-type]


def make_message(study_id: str, minutes: int = 0, **overrides: object) -> ChatMessage:
    fields: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "study_id": study_id,
        "role": "user",
        "content": f"Question at +{minutes}m",
        "created_at": BASE_TIME + timedelta(minutes=minutes),
    }
    fields.update(overrides)
    return ChatMessage(**fields)  # type: ignore[arg-type]
