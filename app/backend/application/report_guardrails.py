"""Rules a drafted report must satisfy before it is stored."""

import re

from application.clinical_report import ClinicalReportOutput
from domain.models import Prediction

_DEFINITIVE = (
    "the patient has",
    "diagnosed with",
    "is diagnosed",
    "the diagnosis is",
    "confirmed diagnosis",
)
_TREATMENT = (
    "prescribe",
    "prescription",
    "medication",
    "dexamethasone",
    "corticosteroid",
    "chemotherapy",
    "radiotherapy",
    "radiation therapy",
    "temozolomide",
    "dosage",
    "start treatment",
    "initiate treatment",
    "treat with",
)
_UNCERTAINTY = ("uncertain", "discrepan", "disagree")
_EXPERT_REVIEW = "expert review"


def find_violations(report: ClinicalReportOutput, prediction: Prediction) -> list[str]:
    """Reasons the report must be rejected; empty when it is acceptable."""
    text = _full_text(report)
    violations = [
        f"definitive diagnosis wording: {phrase!r}"
        for phrase in _DEFINITIVE
        if _contains(text, phrase)
    ]
    violations += [
        f"treatment or medication wording: {term!r}"
        for term in _TREATMENT
        if _contains(text, term)
    ]
    if prediction.needs_expert_triage or report.ai_agreement != "agrees":
        violations += _missing_escalation(report)
    return violations


def _missing_escalation(report: ClinicalReportOutput) -> list[str]:
    narrative = " ".join(
        [report.findings, report.impression, *report.suggested_next_steps]
    ).lower()
    missing = []
    if not any(word in narrative for word in _UNCERTAINTY):
        missing.append("uncertainty or discrepancy is not stated")
    if _EXPERT_REVIEW not in narrative:
        missing.append("expert review is not recommended")
    return missing


def _full_text(report: ClinicalReportOutput) -> str:
    return " ".join(
        [
            report.findings,
            report.visual_observations,
            report.impression,
            report.differential_considerations,
            *report.suggested_next_steps,
        ]
    )


def _contains(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE) is not None
