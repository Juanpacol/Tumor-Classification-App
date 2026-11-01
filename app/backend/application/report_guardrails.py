"""Rules a drafted report must satisfy before it is stored."""

import re

from application.grounded_report import ClaimOutput, GroundedClinicalReportOutput
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


def find_violations(
    report: GroundedClinicalReportOutput, prediction: Prediction
) -> list[str]:
    """Reasons the report must be rejected; empty when it is acceptable."""
    findings = _texts(report.findings)
    impression = _texts(report.impression)
    text = " ".join(
        [
            *findings,
            *_texts(report.visual_observations),
            *impression,
            *_texts(report.differential_considerations),
            *report.suggested_next_steps,
        ]
    )
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
        violations += _missing_escalation(
            [*findings, *impression, *report.suggested_next_steps]
        )
    return violations


def _texts(claims: list[ClaimOutput]) -> list[str]:
    return [claim.text for claim in claims]


def _missing_escalation(narrative: list[str]) -> list[str]:
    joined = " ".join(narrative).lower()
    missing = []
    if not any(word in joined for word in _UNCERTAINTY):
        missing.append("uncertainty or discrepancy is not stated")
    if _EXPERT_REVIEW not in joined:
        missing.append("expert review is not recommended")
    return missing


def _contains(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE) is not None
