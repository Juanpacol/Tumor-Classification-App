from datetime import UTC, datetime

from domain.models import ClinicalReport, Report, ReportStatus

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _content() -> ClinicalReport:
    return ClinicalReport(
        study_id="s1",
        findings="Mass in left frontal lobe",
        visual_observations="Heatmap focused on lesion",
        impression="Consistent with glioma",
        differential_considerations="Metastasis",
        suggested_next_steps=["Contrast MRI"],
        ai_agreement="agree",
    )


def test_pending_report_has_no_content_or_error():
    report = Report(study_id="s1", status=ReportStatus.PENDING, model="m", created_at=NOW)

    assert report.content is None
    assert report.error is None


def test_completed_report_wraps_clinical_content():
    report = Report(
        study_id="s1",
        status=ReportStatus.COMPLETED,
        model="m",
        created_at=NOW,
        content=_content(),
    )

    assert report.content == _content()


def test_failed_report_keeps_error_without_content():
    report = Report(
        study_id="s1",
        status=ReportStatus.FAILED,
        model="m",
        created_at=NOW,
        error="LLM timeout",
    )

    assert report.content is None
    assert report.error == "LLM timeout"
