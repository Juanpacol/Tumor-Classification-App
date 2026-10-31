from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from application.get_study import GetStudy
from application.list_studies import ListStudies, StudyDetail
from application.upload_study import UploadStudy
from domain.errors import DomainError
from domain.models import Prediction, Report
from ports.storage_port import StoragePort
from ports.study_repository_port import StudyRepositoryPort

IMAGE_MEDIA_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}


class PredictionResponse(BaseModel):
    label: str
    probabilities: dict[str, float]
    confidence: float
    needs_expert_triage: bool
    quality: dict[str, Any]
    top_features: list[dict[str, Any]]
    model_version: str


class ClinicalReportResponse(BaseModel):
    findings: str
    visual_observations: str
    impression: str
    differential_considerations: str
    suggested_next_steps: list[str]
    ai_agreement: str


class ReportResponse(BaseModel):
    status: str
    model: str
    created_at: str
    content: ClinicalReportResponse | None
    error: str | None


class StudyDetailResponse(BaseModel):
    id: str
    status: str
    created_at: str
    clinical_notes: str | None
    rejection_reason: str | None
    prediction: PredictionResponse | None
    report: ReportResponse | None


class StudyPageResponse(BaseModel):
    items: list[StudyDetailResponse]
    page: int
    page_size: int
    total: int


def _prediction_response(prediction: Prediction | None) -> PredictionResponse | None:
    if prediction is None:
        return None
    return PredictionResponse(
        label=prediction.label,
        probabilities=prediction.probabilities,
        confidence=prediction.confidence,
        needs_expert_triage=prediction.needs_expert_triage,
        quality=prediction.quality,
        top_features=prediction.top_features,
        model_version=prediction.model_version,
    )


def _report_response(report: Report | None) -> ReportResponse | None:
    if report is None:
        return None
    content = report.content
    return ReportResponse(
        status=report.status.value,
        model=report.model,
        created_at=report.created_at.isoformat(),
        error=report.error,
        content=(
            ClinicalReportResponse(
                findings=content.findings,
                visual_observations=content.visual_observations,
                impression=content.impression,
                differential_considerations=content.differential_considerations,
                suggested_next_steps=content.suggested_next_steps,
                ai_agreement=content.ai_agreement,
            )
            if content
            else None
        ),
    )


def _study_response(detail: StudyDetail) -> StudyDetailResponse:
    study = detail.study
    return StudyDetailResponse(
        id=study.id,
        status=study.status.value,
        created_at=study.created_at.isoformat(),
        clinical_notes=study.clinical_notes,
        rejection_reason=study.rejection_reason,
        prediction=_prediction_response(detail.prediction),
        report=_report_response(detail.report),
    )


def _original_format(file: UploadFile) -> str:
    content_type = file.content_type or ""
    if content_type in IMAGE_MEDIA_TYPES.values():
        return next(k for k, v in IMAGE_MEDIA_TYPES.items() if v == content_type)
    suffix = (file.filename or "").rsplit(".", 1)[-1].lower()
    return suffix or "unknown"


def create_studies_router(
    repository: StudyRepositoryPort,
    storage: StoragePort,
    list_studies: ListStudies,
    get_study: GetStudy,
    upload_study: UploadStudy,
) -> APIRouter:
    router = APIRouter(prefix="/api/studies", tags=["Studies"])

    @router.get(
        "",
        response_model=StudyPageResponse,
        summary="List studies",
        description="Lists studies newest first, with their prediction and report.",
    )
    async def list_endpoint(page: int = 1, page_size: int = 20) -> StudyPageResponse:
        result = await list_studies.execute(page, page_size)
        return StudyPageResponse(
            items=[_study_response(item) for item in result.items],
            page=result.page,
            page_size=result.page_size,
            total=result.total,
        )

    @router.post(
        "",
        response_model=StudyDetailResponse,
        summary="Upload a study",
        description=(
            "Uploads an MRI image, classifies it and returns the resulting study. "
            "A study whose image fails the quality check is returned as rejected."
        ),
        responses={503: {"description": "The classification model is not available."}},
    )
    async def upload_endpoint(
        file: UploadFile = File(...),  # noqa: B008
        clinical_notes: str | None = Form(default=None),
    ) -> StudyDetailResponse:
        image = await file.read()
        detail = await upload_study.execute(
            image=image,
            original_format=_original_format(file),
            clinical_notes=clinical_notes,
        )
        return _study_response(detail)

    @router.get(
        "/{study_id}",
        response_model=StudyDetailResponse,
        summary="Get a study",
        description="Returns one study with its prediction and report.",
        responses={404: {"description": "The study does not exist."}},
    )
    async def get_endpoint(study_id: str) -> StudyDetailResponse:
        detail = await get_study.execute(study_id)
        return _study_response(detail)

    @router.get(
        "/{study_id}/image",
        response_model=bytes,
        summary="Get a study's image",
        description="Returns the uploaded MRI image for a study.",
        responses={404: {"description": "The study or its image does not exist."}},
    )
    async def image_endpoint(study_id: str) -> Response:
        study = await repository.get(study_id)
        if study is None:
            raise DomainError("STUDY_NOT_FOUND", "Study not found.", status_code=404)
        try:
            image = storage.load(study.file_key)
        except KeyError:
            raise DomainError(
                "IMAGE_NOT_FOUND", "The study image is missing.", status_code=404
            ) from None
        media_type = IMAGE_MEDIA_TYPES.get(
            study.original_format, "application/octet-stream"
        )
        return Response(content=image, media_type=media_type)

    return router
