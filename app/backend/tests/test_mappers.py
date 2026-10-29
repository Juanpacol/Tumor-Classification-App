from adapters.outbound.persistence.sqlalchemy import mappers
from domain.models import ReportStatus, StudyStatus
from tests.factories import make_message, make_prediction, make_report, make_study


def test_study_round_trip():
    study = make_study(status=StudyStatus.REJECTED, rejection_reason="Not a brain MRI")

    row = mappers.study_to_row(study)

    assert row.status == "rejected"
    assert mappers.study_to_domain(row) == study


def test_prediction_round_trip_keeps_nested_json():
    prediction = make_prediction(make_study().id)

    assert mappers.prediction_to_domain(mappers.prediction_to_row(prediction)) == prediction


def test_completed_report_content_serialized_as_json_object():
    report = make_report(make_study().id)

    row = mappers.report_to_row(report)

    assert isinstance(row.content, dict)
    assert row.content["suggested_next_steps"] == ["Contrast MRI", "Neurosurgery consult"]
    assert mappers.report_to_domain(row) == report


def test_failed_report_round_trip_with_null_content():
    report = make_report(
        make_study().id, status=ReportStatus.FAILED, content=None, error="LLM timeout"
    )

    row = mappers.report_to_row(report)

    assert row.content is None
    assert mappers.report_to_domain(row) == report


def test_message_round_trip():
    message = make_message(make_study().id, role="assistant")

    assert mappers.message_to_domain(mappers.message_to_row(message)) == message
