import pytest

from application.clinical_report import ClinicalReportOutput
from application.report_guardrails import find_violations
from tests.factories import make_prediction


def report(**overrides: object) -> ClinicalReportOutput:
    fields: dict[str, object] = {
        "findings": "A lesion is consistent with a glioma.",
        "visual_observations": "Heterogeneous signal in the left frontal lobe.",
        "impression": "Suggestive of a glioma. Radiologist confirmation is required.",
        "differential_considerations": "Meningioma cannot be excluded.",
        "suggested_next_steps": ["Contrast-enhanced MRI", "Neurosurgery referral"],
        "ai_agreement": "agrees",
    }
    fields.update(overrides)
    return ClinicalReportOutput(**fields)  # type: ignore[arg-type]


def confident_prediction() -> object:
    return make_prediction("s", needs_expert_triage=False)


def test_a_cautious_report_has_no_violations() -> None:
    assert find_violations(report(), confident_prediction()) == []  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "phrase",
    [
        "The patient has a glioma.",
        "She was diagnosed with a meningioma.",
        "The diagnosis is pituitary adenoma.",
        "This is a confirmed diagnosis.",
        "THE PATIENT HAS a tumor.",
    ],
)
def test_definitive_wording_is_rejected(phrase: str) -> None:
    violations = find_violations(report(impression=phrase), confident_prediction())  # type: ignore[arg-type]

    assert any("definitive diagnosis wording" in v for v in violations)


@pytest.mark.parametrize(
    "step",
    [
        "Start chemotherapy",
        "Prescribe dexamethasone",
        "Radiotherapy planning",
        "Review current medication",
    ],
)
def test_treatment_and_medication_are_rejected(step: str) -> None:
    violations = find_violations(
        report(suggested_next_steps=[step]), confident_prediction()  # type: ignore[arg-type]
    )

    assert any("treatment or medication wording" in v for v in violations)


def test_diagnostic_workup_and_referrals_are_allowed() -> None:
    steps = [
        "Contrast-enhanced MRI",
        "Follow-up imaging in 3 months",
        "Neuro-oncology referral",
        "Multidisciplinary review",
    ]

    assert find_violations(report(suggested_next_steps=steps), confident_prediction()) == []  # type: ignore[arg-type]


def test_a_flagged_study_must_state_uncertainty_and_recommend_expert_review() -> None:
    flagged = make_prediction("s", needs_expert_triage=True)

    violations = find_violations(report(), flagged)

    assert "uncertainty or discrepancy is not stated" in violations
    assert "expert review is not recommended" in violations


def test_a_flagged_study_passes_with_the_escalation_wording() -> None:
    flagged = make_prediction("s", needs_expert_triage=True)
    escalated = report(
        impression="The result is uncertain. Radiologist confirmation is required.",
        suggested_next_steps=["Expert review of the study", "Contrast-enhanced MRI"],
    )

    assert find_violations(escalated, flagged) == []


@pytest.mark.parametrize("agreement", ["disagrees", "uncertain"])
def test_disagreement_requires_the_escalation_even_without_triage(
    agreement: str,
) -> None:
    violations = find_violations(
        report(ai_agreement=agreement), confident_prediction()  # type: ignore[arg-type]
    )

    assert "expert review is not recommended" in violations


def test_escalation_wording_can_be_in_the_next_steps() -> None:
    escalated = report(
        ai_agreement="disagrees",
        findings="The image shows a discrepancy with the classifier.",
        suggested_next_steps=["Expert review", "Contrast-enhanced MRI"],
    )

    assert find_violations(escalated, confident_prediction()) == []  # type: ignore[arg-type]
