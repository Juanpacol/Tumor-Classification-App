"""Rules a drafted report must satisfy before it is stored."""

import re

from application.clinical_report import ClinicalReportOutput
from application.grounded_report import GroundedClinicalReportOutput
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
    narrative = [
        report.findings,
        report.impression,
        *report.suggested_next_steps,
    ]
    return _find_violations(
        text=_full_text(report),
        narrative=narrative,
        ai_agreement=report.ai_agreement,
        prediction=prediction,
    )


def find_violations_grounded(
    report: GroundedClinicalReportOutput, prediction: Prediction
) -> list[str]:
    """Same rules as `find_violations`, for a report whose fields are lists
    of grounded claims instead of free text."""
    findings = [claim.text for claim in report.findings]
    impression = [claim.text for claim in report.impression]
    narrative = [*findings, *impression, *report.suggested_next_steps]
    text = " ".join(
        [
            *findings,
            *(c.text for c in report.visual_observations),
            *impression,
            *(c.text for c in report.differential_considerations),
            *report.suggested_next_steps,
        ]
    )
    return _find_violations(
        text=text,
        narrative=narrative,
        ai_agreement=report.ai_agreement,
        prediction=prediction,
    )


def _find_violations(
    *, text: str, narrative: list[str], ai_agreement: str, prediction: Prediction
) -> list[str]:
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
    if prediction.needs_expert_triage or ai_agreement != "agrees":
        violations += _missing_escalation(narrative)
    return violations


def _missing_escalation(narrative: list[str]) -> list[str]:
    joined = " ".join(narrative).lower()
    missing = []
    if not any(word in joined for word in _UNCERTAINTY):
        missing.append("uncertainty or discrepancy is not stated")
    if _EXPERT_REVIEW not in joined:
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
