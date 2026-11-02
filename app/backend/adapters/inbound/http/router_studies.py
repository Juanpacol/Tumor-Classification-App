from typing import Any
from uuid import UUID

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from application.generate_report import GenerateClinicalReport
from application.get_study import GetStudy
from application.image_upload import MAX_UPLOAD_BYTES
from application.list_studies import ListStudies, StudyDetail
from application.upload_study import StudyRejectedError, UploadStudy
from domain.errors import DomainError
from domain.models import Prediction, Report
from ports.storage_port import StoragePort
from ports.study_repository_port import StudyRepositoryPort

IMAGE_MEDIA_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}
MAX_NOTES_LENGTH = 2000
MAX_PAGE_SIZE = 100
INVALID_ID = {422: {"description": "The study id is not a valid UUID."}}


class PredictionResponse(BaseModel):
    label: str
    probabilities: dict[str, float]
    confidence: float
    needs_expert_triage: bool
    quality: dict[str, Any]
    top_features: list[dict[str, Any]]
    model_version: str


class EvidenceRefResponse(BaseModel):
    key: str
    description: str


class CitationResponse(BaseModel):
    claim: str
    evidence: list[EvidenceRefResponse]
    verified: bool


class ClinicalReportResponse(BaseModel):
    findings: str
    visual_observations: str
    impression: str
    differential_considerations: str
    suggested_next_steps: list[str]
    ai_agreement: str
    citations: list[CitationResponse]


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


class StudyRejectedResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any]
    study: StudyDetailResponse


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
                citations=[
                    CitationResponse(
                        claim=citation.claim,
                        evidence=[
                            EvidenceRefResponse(
                                key=ref.key, description=ref.description
                            )
                            for ref in citation.evidence
                        ],
                        verified=citation.verified,
                    )
                    for citation in content.citations
                ],
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


def _rejection_response(rejection: StudyRejectedError) -> JSONResponse:
    body = StudyRejectedResponse(
        code=rejection.code,
        message=rejection.message,
        details=rejection.details or {},
        study=_study_response(rejection.detail),
    )
    return JSONResponse(status_code=422, content=body.model_dump())


def create_studies_router(
    repository: StudyRepositoryPort,
    storage: StoragePort,
    list_studies: ListStudies,
    get_study: GetStudy,
    upload_study: UploadStudy,
    generate_report: GenerateClinicalReport,
) -> APIRouter:
    router = APIRouter(prefix="/api/studies", tags=["Studies"])

    @router.get(
        "",
        response_model=StudyPageResponse,
        summary="List studies",
        description=(
            "Lists studies newest first, with their prediction and report. "
            f"Pages start at 1 and hold up to {MAX_PAGE_SIZE} studies."
        ),
        responses={422: {"description": "The page or page size is out of range."}},
    )
    async def list_endpoint(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=MAX_PAGE_SIZE),
    ) -> StudyPageResponse:
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
        status_code=201,
        summary="Upload a study",
        description=(
            "Uploads a JPG or PNG MRI image with optional clinical notes, classifies "
            "it and returns the completed study with its prediction. The format is "
            "read from the content, and the image is stored without its embedded "
            "metadata (EXIF, XMP, text chunks) and without its file name. Nothing is "
            "stored when the request is refused (413, 415, 503 or invalid fields). "
            "A study whose image the classifier rejects is stored as rejected and "
            "returned in a 422; one whose classification fails unexpectedly is "
            "stored as failed and answered with a generic 500."
        ),
        responses={
            413: {"description": "The file is larger than 20 MB."},
            415: {
                "description": (
                    "The file is not a decodable JPG or PNG image (DICOM, other "
                    "formats and corrupt files)."
                )
            },
            422: {
                "model": StudyRejectedResponse,
                "description": (
                    "The classifier rejected the image (`QUALITY_CHECK_FAILED` or "
                    "`UNSUPPORTED_IMAGE`): the body carries the reason and the "
                    "study stored as rejected. A request with no file or with "
                    f"clinical notes longer than {MAX_NOTES_LENGTH} characters is "
                    "also answered with 422, as a validation error with `detail`."
                ),
            },
            500: {
                "description": "Classification failed; the study is stored as failed."
            },
            503: {"description": "The classification model is not available."},
        },
    )
    async def upload_endpoint(
        file: UploadFile = File(...),  # noqa: B008
        clinical_notes: str | None = Form(default=None, max_length=MAX_NOTES_LENGTH),
    ) -> StudyDetailResponse | JSONResponse:
        image = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            detail = await upload_study.execute(
                image=image, clinical_notes=clinical_notes
            )
        except StudyRejectedError as rejection:
            return _rejection_response(rejection)
        return _study_response(detail)

    @router.get(
        "/{study_id}",
        response_model=StudyDetailResponse,
        summary="Get a study",
        description="Returns one study with its prediction and report.",
        responses={404: {"description": "The study does not exist."}, **INVALID_ID},
    )
    async def get_endpoint(study_id: UUID) -> StudyDetailResponse:
        detail = await get_study.execute(str(study_id))
        return _study_response(detail)

    @router.get(
        "/{study_id}/image",
        response_model=bytes,
        summary="Get a study's image",
        description="Returns the uploaded MRI image for a study.",
        responses={
            404: {"description": "The study or its image does not exist."},
            **INVALID_ID,
        },
    )
    async def image_endpoint(study_id: UUID) -> Response:
        study = await repository.get(str(study_id))
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

    @router.post(
        "/{study_id}/report",
        response_model=ReportResponse,
        summary="Generate a study's clinical report",
        description=(
            "Returns the stored report of a completed study, or drafts one with the "
            "LLM. Every claim carries its citations, checked against the study's "
            "evidence. A previously failed report is generated again."
        ),
        responses={
            404: {"description": "The study does not exist."},
            409: {"description": "The study is not completed or has no prediction."},
            502: {"description": "The model did not produce a valid report."},
            503: {"description": "The report service is not available."},
        },
    )
    async def report_endpoint(study_id: str) -> ReportResponse:
        report = await generate_report.execute(study_id)
        response = _report_response(report)
        assert response is not None
        return response

    return router
