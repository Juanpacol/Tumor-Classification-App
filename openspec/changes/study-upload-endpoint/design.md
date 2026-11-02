# Design

## Context

See proposal.md for the motivation and specs/ for the required behavior.

- `origin/AB-118-upload-limits-metadata` (commits `8770bcc`, `083fc9d`) adds `application/image_upload.py:prepare_upload`. It enforces the 20 MB limit (`FileTooLargeError`, 413) and checks the JPEG marker and PNG chunk structure (`UnsupportedMediaTypeError`, 415). It removes APP1/APP13/COM and `eXIf`/`tEXt`/`zTXt`/`iTXt`/`tIME` without re-encoding. It also adds `Form(max_length=2000)` for the notes, removes `_original_format`, and saves FAILED on any non-422 classifier error before re-raising.
- `UploadStudy` stores the image and a PENDING study before classifying. Today it turns *any* 422 `DomainError` into REJECTED, using the generic `error.message` as the reason.
- The ml_engine's `decode_mri_image` decodes with Pillow and raises `UnsupportedImageError` (422) for DICOM, non-JPEG/PNG data, corrupt data, under 64 px, over 25 M pixels and over 50 MB. It also applies EXIF orientation.
- `register_exception_handlers` only handles `DomainError`. When an `Exception` handler is registered, Starlette's `ServerErrorMiddleware` still re-raises after responding, so uvicorn logs the full traceback, exception message included.
- The portal's `httpClient.send` throws `errorFromResponse` for any non-2xx response. A FastAPI validation 422 (`{detail: [...]}`) becomes `ApiError(UNKNOWN_ERROR, ..., 422)`. `uploadChecks.ts` maps statuses to messages and already blocks notes over 2000 characters before sending.
- `tests/test_architecture_boundaries.py` keeps `domain/` free of application and framework imports.

## Goals / Non-Goals

**Goals:**
- Every outcome of the upload has exactly one status in storage and one HTTP response, as listed in `specs/study-upload`.
- Nothing is written for refusals (422 validation, 413, 415, 503).
- No PHI reaches the logs, including through tracebacks.

**Non-Goals:**
- Cleaning up storage when the repository write fails after the image was saved.
- Asynchronous classification and the PROCESSING status.
- Restoring EXIF orientation after stripping (see Risks).

## Decisions

### D1. Rebase onto the teammate branch and keep its stripper
Rebase `AB-118-IMG-study-upload` onto `origin/AB-118-upload-limits-metadata` and extend `prepare_upload` instead of replacing it. Its approach of stripping without re-encoding keeps the classified pixels identical to the uploaded ones, which a Pillow re-save would not guarantee for JPEG.
*Alternative*: re-implement on this branch. Rejected because it duplicates reviewed work and guarantees conflicts.

### D2. Decode the pixels in `prepare_upload` after the structure check
After stripping, open the stripped bytes with Pillow (`formats=["JPEG", "PNG"]`) and call `load()`. `OSError`, `SyntaxError`, `ValueError` and `Image.DecompressionBombError` become `UnsupportedMediaTypeError` (415). The decode runs in a worker thread (`anyio.to_thread.run_sync`) so a 20 MB decode does not block the event loop. Decoding the *stripped* bytes proves the bytes we store and classify are readable.
The model's own limits (64 px minimum, 25 M pixels maximum) stay in the ml_engine. Such an image is readable, so it reaches the classifier, gets `UNSUPPORTED_IMAGE` and is REJECTED (422), as the spec requires. The backend does not copy those constants.
*Alternatives*: (B) map the classifier's `UnsupportedImageError` to 415 and delete the stored file afterwards, which needs a delete on `StoragePort` and a rollback path; (C) only check the structure, which leaves corrupt image data persisted as REJECTED. Both rejected.

### D3. Check readiness before anything is stored
`UploadStudy.execute` first calls `classifier.is_ready()` and raises `ModelUnavailableError` (503) if it returns false, before `prepare_upload` or any write. This removes the teammate's FAILED-on-503 path for the normal case. The model loads once at startup and never unloads, so a 503 halfway through classification is not expected. If one happens anyway, it follows D5.

### D4. Rejection is an application-level exception that carries the study
Add `StudyRejectedError(DomainError)` in `application/upload_study.py`, not in `domain/`, because it holds a `StudyDetail` (an application type) and `domain/` must not import `application/`. It copies `code`, `message`, `details` and `status_code=422` from the classifier's `QualityCheckFailedError`/`UnsupportedImageError`. The use case catches exactly those two types, not "any 422". It saves REJECTED with `rejection_reason = details["reason"]` and raises `StudyRejectedError(detail, cause)`.
The router catches `StudyRejectedError` around `upload_study.execute` and returns `JSONResponse(422, {code, message, details, study: _study_response(detail)})`. Building the response in the router keeps the HTTP shape next to `StudyDetailResponse`. A new `StudyRejectedResponse` Pydantic model documents it in `responses={422: {"model": ...}}`. Because FastAPI's notes-validation 422 shares the code, the 422 description names both bodies.
*Alternatives*: return a `StudyDetail` and let the router pick 422 from `study.status`, which hides the classifier's `code`/`details` the spec requires; a global handler for `StudyRejectedError` in `error_handlers.py`, which would pull the study serializers into the error module.

### D5. Unexpected errors become a handled 500 DomainError
In `UploadStudy`, a classifier exception that is not a rejection error does the following:
- A `DomainError` (for example the unexpected 503) saves FAILED and is re-raised unchanged.
- Any other `Exception` saves FAILED, logs one line, and raises `ClassificationFailedError` (`domain/errors.py`, code `INTERNAL_ERROR`, status 500, fixed generic message) `from None`.

The log line is `logger.error("study %s classification failed: %s at %s:%d", study.id, type(exc).__name__, filename, lineno)`. The filename and line come from the last frame of `traceback.extract_tb`. This keeps a debugging pointer without the exception message, which could contain PHI. `exc_info` is never passed.
Since `ClassificationFailedError` is a `DomainError`, the existing handler answers it and Starlette never logs a traceback.
Also register a catch-all `Exception` handler in `error_handlers.py` that returns the same generic `INTERNAL_ERROR` body, so other unexpected failures do not answer in plain text. Its traceback is still logged by Starlette. That is acceptable outside the upload path, where no notes are in scope.
*Alternative*: rely only on the catch-all handler, which still logs the exception message through Starlette's re-raise. Rejected.

### D6. 201 on success
`@router.post(..., status_code=201)`. `response_model` stays `StudyDetailResponse`.

### D7. Portal: treat a rejection 422 as a returned study
In `httpClient.uploadStudy`, call `fetch` through a small variant of `send`. On 422, parse the body: when it has a `study` object, return `body.study` as a `StudyDetail`. Otherwise build the error with the existing `errorFromResponse` logic. `UploadStudy.tsx` then calls `onUploaded(study.id)` exactly as for a completed study, and the dashboard already shows rejected studies with their reason. Add `422: "The clinical notes are longer than 2000 characters."` to `MESSAGE_BY_STATUS`. Only the FastAPI validation 422 reaches it, and the client-side check normally stops long notes before sending. `mockClient` already returns study objects and needs no change.

## Risks / Trade-offs

- [EXIF orientation is lost] Stripping APP1 removes the orientation tag. Before stripping, the ml_engine's `exif_transpose` would have rotated the image, so a phone photo of a film could be classified rotated. → MRI exports rarely carry orientation. Recorded here; revisit if real uploads show it.
- [Double decode] The backend and the ml_engine both decode the image. → At most 20 MB, a few hundred ms, done in a worker thread. This is the price of keeping 415 accurate without a storage rollback.
- [Memory on huge images] A 20 MB PNG may expand to many pixels before the ml_engine's 25 M limit applies. → Pillow's `DecompressionBombError` (about 179 M pixels) caps it and returns 415. Images up to that size decode in worker memory, which Cloud Run's instance memory covers.
- [Orphaned image] If `repository.save` fails after `storage.save`, the file remains under `studies/{uuid}`. → Out of scope (see Non-Goals). The key has no PHI.
- [Less debugging detail] The upload failure log omits the exception message. → It keeps the type and raising frame. Reproduce locally with the same image if needed.
- [Breaking status codes] An older portal build against the new backend shows a rejection as an error, and 201 is still `ok`. → Backend and frontend ship in the same PR and deploy together.

## Migration Plan

No schema change. Deploy the backend (Cloud Run) and frontend (Vercel) from the same merge. Rollback means redeploying the previous revision of both.
