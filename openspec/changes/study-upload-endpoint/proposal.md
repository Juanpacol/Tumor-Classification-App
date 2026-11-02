# Proposal

## Why

Clinicians need to upload a brain MRI image with optional notes and get the AI classification back as a second opinion (AB-118). `POST /api/studies` exists, but it stores whatever bytes arrive, EXIF metadata included, without a size limit. It reports a quality rejection as a success. An unexpected classifier error leaves the study stuck in PENDING and returns Starlette's plain-text 500. A teammate's branch (`origin/AB-118-upload-limits-metadata`) adds the 413/415 limits, the notes limit and metadata stripping. The status codes, failure handling and log hygiene from the acceptance criteria are still missing.

## What Changes

- Build on `origin/AB-118-upload-limits-metadata` (20 MB limit with 413; JPEG/PNG detected from the content with 415; EXIF/XMP/IPTC/comments/PNG text removed without re-encoding; the filename is not stored; notes over 2000 characters return 422) and keep its behavior.
- Fully decode the pixels before anything is stored, so a file with a valid structure but corrupt image data gets 415 and nothing is persisted.
- Refuse an upload with 503 before anything is stored when no model is loaded.
- **BREAKING**: a successful upload returns **201** instead of 200.
- **BREAKING**: a rejected study (failed quality gate, or an image too small or too large for the model) returns **422** with `code`, `message`, `details` and the rejected `study` in its body, instead of a 200. The study is saved as REJECTED with the specific reason (`details.reason`), not the generic message.
- An unexpected classifier error saves the study as FAILED and returns 500 with the generic body `{"code": "INTERNAL_ERROR", "message": ...}`. The log entry holds only the study id and the exception type, never notes, filenames or image content.
- Portal: the HTTP client treats a `QUALITY_CHECK_FAILED`/`UNSUPPORTED_IMAGE` 422 as a returned study, so the rejected study opens just like a completed one. The notes-validation 422 gets a plain-language message.
- Out of scope: automatic PHI scrubbing of free-text clinical notes (the UI warns users instead).

## Capabilities

### New Capabilities
- `study-upload`: accepting an MRI upload over HTTP, covering input limits, content-based format checks, metadata-free storage keyed by study UUID, synchronous classification, and the status and HTTP response for each outcome (completed, rejected, failed, refused). Also covers keeping PHI out of the logs.

### Modified Capabilities
- `mri-classification`: "Unusable images are rejected with their reason" changes. Files that cannot be decoded are now refused at upload with 415 before classification, so the 422 `UNSUPPORTED_IMAGE` path only covers decodable images the model cannot use (for example, outside its size limits), and an upload's 422 also carries the rejected study.

## Impact

- **Backend** (`app/backend`): `adapters/inbound/http/router_studies.py` (201, 422 body, documented responses), `adapters/inbound/http/error_handlers.py` (generic 500 handler), `application/upload_study.py` (readiness check, outcome handling, reason), `application/image_upload.py` (from the teammate branch, plus the pixel decode), `domain/errors.py` (rejection error that carries the study), tests (`test_upload_study.py`, `test_studies_router.py`, `test_image_upload.py`, `test_api_documentation.py`).
- **Dependencies**: `pillow` becomes a direct backend dependency in `pyproject.toml`. It is already installed through `ml_engine`.
- **Frontend** (`app/frontend/src`): `services/api/httpClient.ts`, `features/portal/uploadChecks.ts`, and their tests. The mock client already returns study objects and needs no change.
- **API clients**: callers that relied on 200 for success or on a 200 rejected body must handle 201 and 422. The portal is the only client.
- **Git**: `AB-118-IMG-study-upload` is rebased onto `origin/AB-118-upload-limits-metadata` before the new work begins.
