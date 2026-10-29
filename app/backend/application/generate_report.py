"""Use case: draft the clinical report of a completed study with the LLM."""

from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import ValidationError

from application.clinical_report import ClinicalReportOutput
from application.report_guardrails import find_violations
from application.report_prompt import build_messages
from domain.errors import DomainError
from domain.models import (
    ClinicalReport,
    Prediction,
    Report,
    ReportStatus,
    Study,
    StudyStatus,
)
from ports.llm_port import LLMPort
from ports.storage_port import StoragePort
from ports.study_repository_port import StudyRepositoryPort

MAX_ATTEMPTS = 2


class GenerateClinicalReport:
    """Returns the stored report, or asks the LLM for one and validates it."""

    def __init__(
        self,
        repository: StudyRepositoryPort,
        storage: StoragePort,
        llm: LLMPort,
        model: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._llm = llm
        self._model = model
        self._clock = clock

    async def execute(self, study_id: str, regenerate: bool = False) -> Report:
        """Report of a COMPLETED study.

        Returns the stored report unless ``regenerate`` is set or the last one
        failed. Raises ``DomainError`` 404 for an unknown study, 409 for a study
        that is not completed and 502 when the LLM output stays invalid after one
        retry (the failure is stored and generation can be tried again).
        """
        study = await self._require_study(study_id)
        existing = await self._repository.get_report(study_id)
        if existing and existing.status is ReportStatus.COMPLETED and not regenerate:
            return existing

        prediction = await self._require_prediction(study_id)
        image = self._load_image(study)
        output, problems = await self._draft(study, prediction, image)
        if output is None:
            await self._store(study_id, ReportStatus.FAILED, error="; ".join(problems))
            raise DomainError(
                "REPORT_INVALID_OUTPUT",
                "The model did not produce a valid report. Try again.",
                status_code=502,
            )
        report = ClinicalReport(study_id=study_id, **output.model_dump())
        return await self._store(study_id, ReportStatus.COMPLETED, content=report)

    async def _require_study(self, study_id: str) -> Study:
        study = await self._repository.get(study_id)
        if study is None:
            raise DomainError("STUDY_NOT_FOUND", "Study not found.", status_code=404)
        if study.status is not StudyStatus.COMPLETED:
            raise DomainError(
                "STUDY_NOT_COMPLETED",
                f"A report needs a completed study, this one is {study.status.value}.",
                status_code=409,
            )
        return study

    async def _require_prediction(self, study_id: str) -> Prediction:
        prediction = await self._repository.get_prediction(study_id)
        if prediction is None:
            raise DomainError(
                "PREDICTION_NOT_FOUND",
                "The study has no prediction to report on.",
                status_code=409,
            )
        return prediction

    def _load_image(self, study: Study) -> bytes:
        try:
            return self._storage.load(study.file_key)
        except KeyError:
            raise DomainError(
                "IMAGE_NOT_FOUND", "The study image is missing.", status_code=409
            ) from None

    async def _draft(
        self, study: Study, prediction: Prediction, image: bytes
    ) -> tuple[ClinicalReportOutput | None, list[str]]:
        messages = build_messages(study, prediction)
        problems: list[str] = []
        for _ in range(MAX_ATTEMPTS):
            output: ClinicalReportOutput | None = None
            try:
                output = await self._llm.complete_json(
                    messages, ClinicalReportOutput, image=image
                )
                problems = find_violations(output, prediction)
            except ValidationError as error:
                problems = [
                    f"output does not match the schema: {error.error_count()} errors"
                ]
            if output is not None and not problems:
                return output, []
            messages = [
                *build_messages(study, prediction),
                {
                    "role": "user",
                    "content": "Your previous answer was rejected: "
                    + "; ".join(problems)
                    + ". Return a corrected report.",
                },
            ]
        return None, problems

    async def _store(
        self,
        study_id: str,
        status: ReportStatus,
        content: ClinicalReport | None = None,
        error: str | None = None,
    ) -> Report:
        report = Report(
            study_id=study_id,
            status=status,
            model=self._model,
            created_at=self._clock(),
            content=content,
            error=error,
        )
        await self._repository.save_report(report)
        return report
