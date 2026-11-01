from datetime import UTC, datetime
from typing import Any

import pytest

from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from adapters.outbound.storage.in_memory_storage import InMemoryStorage
from application.generate_report import MAX_ATTEMPTS, GenerateClinicalReport
from application.grounded_report import ClaimOutput, GroundedClinicalReportOutput
from domain.errors import DomainError
from domain.models import Report, ReportStatus, StudyStatus
from tests.doubles import ScriptedLLM
from tests.factories import make_prediction, make_study

IMAGE = b"de-identified-png"
NOW = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)


NARRATIVE_FIELDS = (
    "findings",
    "visual_observations",
    "impression",
    "differential_considerations",
)


def claim(text: str, *keys: str) -> ClaimOutput:
    return ClaimOutput(text=text, evidence_keys=list(keys))


def valid_output(**overrides: Any) -> GroundedClinicalReportOutput:
    """Narrative overrides are plain text; each becomes a single claim."""
    narrative: dict[str, Any] = {
        "findings": claim("A lesion consistent with a glioma.", "prediction.label"),
        "visual_observations": claim("Heterogeneous signal in the left frontal lobe."),
        "impression": claim(
            "Suggestive of a glioma. Radiologist confirmation is required."
        ),
        "differential_considerations": claim("Meningioma cannot be excluded."),
    }
    other: dict[str, Any] = {
        "suggested_next_steps": ["Contrast-enhanced MRI", "Neurosurgery referral"],
        "ai_agreement": "agrees",
    }
    for key, value in overrides.items():
        if key in NARRATIVE_FIELDS:
            narrative[key] = value if isinstance(value, ClaimOutput) else claim(value)
        else:
            other[key] = value
    return GroundedClinicalReportOutput(
        **{key: [value] for key, value in narrative.items()}, **other
    )


def definitive() -> GroundedClinicalReportOutput:
    return valid_output(impression="The patient has a glioma.")


def escalated(**overrides: Any) -> GroundedClinicalReportOutput:
    return valid_output(
        impression="The result is uncertain. Radiologist confirmation is required.",
        suggested_next_steps=["Expert review of the study", "Contrast-enhanced MRI"],
        **overrides,
    )


class Setup:
    def __init__(self, llm: ScriptedLLM, **study: Any) -> None:
        self.llm = llm
        self.repository = InMemoryStudyRepository()
        self.storage = InMemoryStorage()
        self.study = make_study(status=StudyStatus.COMPLETED, **study)
        self.use_case = GenerateClinicalReport(
            self.repository,
            self.storage,
            llm,
            model="muse-spark-1.3",
            clock=lambda: NOW,
        )

    async def prepare(self, triage: bool = False, with_image: bool = True) -> None:
        await self.repository.save(self.study)
        await self.repository.save_prediction(
            make_prediction(self.study.id, needs_expert_triage=triage)
        )
        if with_image:
            self.storage.save(IMAGE, self.study.file_key)


async def ready(*responses: Any, triage: bool = False, **study: Any) -> Setup:
    setup = Setup(ScriptedLLM(*responses), **study)
    await setup.prepare(triage=triage)
    return setup


async def test_a_valid_report_is_stored_and_returned() -> None:
    setup = await ready(valid_output())

    report = await setup.use_case.execute(setup.study.id)

    assert report.status is ReportStatus.COMPLETED
    assert report.model == "muse-spark-1.3" and report.created_at == NOW
    assert report.content is not None and report.content.ai_agreement == "agrees"
    assert report.content.suggested_next_steps == [
        "Contrast-enhanced MRI",
        "Neurosurgery referral",
    ]
    assert await setup.repository.get_report(setup.study.id) == report


async def test_the_model_receives_the_image_the_prompt_and_the_schema() -> None:
    setup = await ready(valid_output(), clinical_notes="Headache for 3 weeks")

    await setup.use_case.execute(setup.study.id)

    (call,) = setup.llm.calls
    assert call["image"] == IMAGE and call["schema"] is GroundedClinicalReportOutput
    assert call["messages"][0]["role"] == "system"
    assert "Headache for 3 weeks" in call["messages"][1]["content"]
    assert "prediction.label: glioma" in call["messages"][1]["content"]


async def test_the_claims_become_the_report_text() -> None:
    output = valid_output()
    output.findings.append(claim("A second finding."))
    setup = await ready(output)

    report = await setup.use_case.execute(setup.study.id)

    assert report.content is not None
    assert report.content.findings == (
        "A lesion consistent with a glioma. A second finding."
    )


async def test_a_claim_grounded_in_the_bundle_is_stored_as_verified() -> None:
    setup = await ready(valid_output())

    report = await setup.use_case.execute(setup.study.id)

    assert report.content is not None
    grounded = next(
        c for c in report.content.citations if c.claim.startswith("A lesion")
    )
    assert grounded.verified is True
    assert [ref.key for ref in grounded.evidence] == ["prediction.label"]


async def test_a_claim_citing_an_invented_key_is_stored_as_unverified() -> None:
    invented = valid_output(
        findings=claim("The lesion is 3cm wide.", "prediction.tumor_size")
    )
    setup = await ready(invented)

    report = await setup.use_case.execute(setup.study.id)

    assert report.content is not None
    ungrounded = next(c for c in report.content.citations if "3cm" in c.claim)
    assert ungrounded.verified is False


async def test_a_claim_without_evidence_is_kept_but_unverified() -> None:
    setup = await ready(valid_output())

    report = await setup.use_case.execute(setup.study.id)

    assert report.content is not None
    visual = next(c for c in report.content.citations if "Heterogeneous" in c.claim)
    assert visual.verified is False and visual.evidence == []


async def test_an_existing_report_is_returned_without_calling_the_model() -> None:
    setup = await ready(valid_output())
    first = await setup.use_case.execute(setup.study.id)

    second = await setup.use_case.execute(setup.study.id)

    assert second == first and len(setup.llm.calls) == 1


async def test_regenerate_asks_the_model_again_and_replaces_the_report() -> None:
    setup = await ready(valid_output(), valid_output(findings="Second draft."))
    await setup.use_case.execute(setup.study.id)

    report = await setup.use_case.execute(setup.study.id, regenerate=True)

    assert report.content is not None and report.content.findings == "Second draft."
    assert len(setup.llm.calls) == 2


async def test_a_failed_report_is_generated_again_without_asking() -> None:
    setup = await ready(valid_output())
    await setup.repository.save_report(
        Report(setup.study.id, ReportStatus.FAILED, "m", NOW, error="old failure")
    )

    report = await setup.use_case.execute(setup.study.id)

    assert report.status is ReportStatus.COMPLETED


async def test_invalid_output_is_retried_once_with_the_reasons() -> None:
    setup = await ready(definitive(), valid_output())

    report = await setup.use_case.execute(setup.study.id)

    assert report.status is ReportStatus.COMPLETED
    assert len(setup.llm.calls) == 2
    feedback = setup.llm.calls[1]["messages"][-1]["content"]
    assert "rejected" in feedback and "definitive diagnosis wording" in feedback


async def test_two_invalid_outputs_fail_the_report_and_answer_502() -> None:
    setup = await ready(definitive(), definitive())

    with pytest.raises(DomainError) as error:
        await setup.use_case.execute(setup.study.id)

    assert (
        error.value.status_code == 502 and error.value.code == "REPORT_INVALID_OUTPUT"
    )
    assert len(setup.llm.calls) == MAX_ATTEMPTS
    stored = await setup.repository.get_report(setup.study.id)
    assert stored is not None and stored.status is ReportStatus.FAILED
    assert stored.content is None and "the patient has" in (stored.error or "")


async def test_after_a_failure_generation_can_be_retried() -> None:
    setup = await ready(definitive(), definitive(), valid_output())
    with pytest.raises(DomainError):
        await setup.use_case.execute(setup.study.id)

    report = await setup.use_case.execute(setup.study.id)

    assert report.status is ReportStatus.COMPLETED


async def test_output_that_breaks_the_schema_counts_as_invalid() -> None:
    wrong = {**valid_output().model_dump(), "ai_agreement": "maybe"}
    setup = await ready(wrong, valid_output())

    report = await setup.use_case.execute(setup.study.id)

    assert report.status is ReportStatus.COMPLETED and len(setup.llm.calls) == 2


async def test_a_treatment_recommendation_is_invalid() -> None:
    treatment = valid_output(suggested_next_steps=["Start chemotherapy"])
    setup = await ready(treatment, treatment)

    with pytest.raises(DomainError):
        await setup.use_case.execute(setup.study.id)


async def test_a_flagged_study_needs_the_uncertainty_and_expert_review_wording() -> (
    None
):
    setup = await ready(valid_output(), escalated(), triage=True)

    report = await setup.use_case.execute(setup.study.id)

    assert report.content is not None
    assert "Expert review" in report.content.suggested_next_steps[0]
    assert len(setup.llm.calls) == 2


async def test_a_disagreement_needs_the_escalation_wording() -> None:
    setup = await ready(
        valid_output(ai_agreement="disagrees"), escalated(ai_agreement="disagrees")
    )

    report = await setup.use_case.execute(setup.study.id)

    assert report.content is not None and report.content.ai_agreement == "disagrees"
    assert len(setup.llm.calls) == 2


async def test_an_unknown_study_is_a_404() -> None:
    setup = Setup(ScriptedLLM())

    with pytest.raises(DomainError) as error:
        await setup.use_case.execute("missing")

    assert error.value.status_code == 404 and error.value.code == "STUDY_NOT_FOUND"


@pytest.mark.parametrize(
    "status",
    [
        StudyStatus.REJECTED,
        StudyStatus.FAILED,
        StudyStatus.PENDING,
        StudyStatus.PROCESSING,
    ],
)
async def test_a_study_that_is_not_completed_is_a_409_and_the_model_is_not_called(
    status: StudyStatus,
) -> None:
    setup = Setup(ScriptedLLM())
    setup.study.status = status
    await setup.prepare()

    with pytest.raises(DomainError) as error:
        await setup.use_case.execute(setup.study.id)

    assert error.value.status_code == 409 and error.value.code == "STUDY_NOT_COMPLETED"
    assert setup.llm.calls == []


async def test_a_study_without_prediction_is_a_409() -> None:
    setup = Setup(ScriptedLLM())
    await setup.repository.save(setup.study)

    with pytest.raises(DomainError) as error:
        await setup.use_case.execute(setup.study.id)

    assert error.value.code == "PREDICTION_NOT_FOUND" and error.value.status_code == 409


async def test_a_study_without_image_is_a_409() -> None:
    setup = Setup(ScriptedLLM())
    await setup.prepare(with_image=False)

    with pytest.raises(DomainError) as error:
        await setup.use_case.execute(setup.study.id)

    assert error.value.code == "IMAGE_NOT_FOUND" and error.value.status_code == 409


async def test_an_unavailable_llm_is_passed_on_and_nothing_is_stored() -> None:
    unavailable = DomainError("LLM_UNAVAILABLE", "No key.", status_code=503)
    setup = await ready(unavailable)

    with pytest.raises(DomainError) as error:
        await setup.use_case.execute(setup.study.id)

    assert error.value.status_code == 503
    assert await setup.repository.get_report(setup.study.id) is None
