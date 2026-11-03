# Tasks

## 1. Base branch

- [x] 1.1 Bring the teammate's work from `origin/AB-118-upload-limits-metadata` into `AB-118-IMG-study-upload` by cherry-picking `8770bcc` onto the develop-based branch (its merge commit `083fc9d` is skipped and its conflict resolution reused, so develop commits keep their SHAs), keeping the uncommitted `app/backend/.dockerignore` and this change's `openspec/` files. Verify `git log` shows the teammate's commit and `cd app/backend && python -m pytest tests/ -v` passes
- [x] 1.2 Add `pillow` to `app/backend/pyproject.toml` dependencies and verify `python -c "import PIL"` works in the backend environment

## 2. Upload refusals: decode check and readiness (design D2, D3)

- [x] 2.1 Extend `prepare_upload` in `application/image_upload.py` to decode the stripped bytes with Pillow (JPEG/PNG only), mapping decode errors and `DecompressionBombError` to `UnsupportedMediaTypeError`. Verify with new `tests/test_image_upload.py` cases: truncated JPEG data and a corrupt PNG `IDAT` both raise, and a valid image passes
- [x] 2.2 Run the decode off the event loop in `UploadStudy.execute` (`anyio.to_thread.run_sync`) and verify the existing upload tests still pass
- [x] 2.3 Make `UploadStudy.execute` raise `ModelUnavailableError` when `classifier.is_ready()` is false, before preparing or storing anything. Add `ready` control to `ScriptedClassifier` in `tests/doubles.py` and verify a test asserts the 503 and empty storage and repository
- [x] 2.4 Add router tests in `tests/test_studies_router.py` for every refusal (a 2001-character note and a missing file give 422; 20 MB + 1 byte gives 413; DICOM, a text file renamed `.png` and truncated JPEG data give 415; no model gives 503). Each asserts nothing was stored in the repository or storage; verify they pass

## 3. Outcomes: completed, rejected, failed (design D4, D5, D6)

- [x] 3.1 Return 201 from `POST /api/studies` and verify the updated `test_upload_study_returns_a_completed_study` asserts 201, `status: "completed"` and the prediction
- [x] 3.2 Add `StudyRejectedError` in `application/upload_study.py`. Catch only `QualityCheckFailedError`/`UnsupportedImageError`, save REJECTED with `details["reason"]` and raise it. Verify `tests/test_upload_study.py` asserts the stored rejection reason is the specific reason ("too noisy"), not the generic message
- [x] 3.3 Catch `StudyRejectedError` in the router and return 422 `{code, message, details, study}`. Add a `StudyRejectedResponse` model to `responses={422: ...}` (the description names both 422 bodies). Verify router tests for `QUALITY_CHECK_FAILED` and `UNSUPPORTED_IMAGE` assert status 422, `code`, `details.reason`, `study.status == "rejected"` and the stored study
- [x] 3.4 Add `ClassificationFailedError` (`INTERNAL_ERROR`, 500, generic message) to `domain/errors.py`. In `UploadStudy`, any non-`DomainError` from the classifier saves FAILED, logs only the study id, error type and raising frame, and raises it `from None`; a non-rejection `DomainError` saves FAILED and re-raises. Verify use-case tests cover both
- [x] 3.5 Register a catch-all `Exception` handler in `adapters/inbound/http/error_handlers.py` returning the generic `INTERNAL_ERROR` JSON, and verify `tests/test_error_mapping.py` asserts a JSON 500 body for an unexpected error
- [x] 3.6 Add a router test where the classifier raises `RuntimeError("boom at /data/jane_roe")` for an upload with notes "Patient Jane Roe, 54" and filename `jane_roe.png`. Assert 500 with `INTERNAL_ERROR`, a body without "boom"/"jane_roe", a stored study that is `failed`, and (with `caplog`) logs that contain the study id and `RuntimeError` but not "Jane Roe", "jane_roe" or "boom"; verify it passes
- [x] 3.7 Update the upload endpoint's `description` and `responses` (201, 413, 415, 422 both bodies, 500, 503) and verify `tests/test_api_documentation.py` passes and `/docs` shows them

## 4. Metadata-free storage

- [x] 4.1 Add router tests that upload a JPEG with EXIF (camera model, date) and a PNG with `tEXt`/`iTXt`/`eXIf` chunks, built with Pillow in `tests/factories.py`. Assert the stored file's `Image.open(...).getexif()` is empty, the PNG has no text chunks (`.text` empty), the decoded pixels equal the upload's, and the storage key is `studies/<uuid>`; verify they pass
- [x] 4.2 Add a test that uploads `john_doe_mri.png` and asserts "john_doe_mri" appears in neither the storage key, the stored study nor the response; verify it passes

## 5. Portal (design D7)

- [x] 5.1 Make `httpClient.uploadStudy` in `app/frontend/src/services/api/httpClient.ts` return `body.study` for a 422 whose body has a `study`, and otherwise throw as before. Verify `httpClient.test.ts` covers a rejection 422 returning the study and a validation 422 throwing `ApiError` with status 422
- [x] 5.2 Add a 422 message for clinical notes over 2000 characters to `MESSAGE_BY_STATUS` in `features/portal/uploadChecks.ts`. Verify `UploadStudy.test.tsx` shows it, and that a rejected upload calls `onUploaded` with the study id
- [x] 5.3 Verify `npm test`, `npm run typecheck`, `npm run lint` and `npm run format:check` pass in `app/frontend`

## 6. Documentation and integration

- [x] 6.1 Update `app/backend/README.md` (upload statuses and limits; remove the note that the report endpoint is waiting on AB-118's upload) and verify the documented status codes match the endpoint's `responses`
- [x] 6.2 Run the whole backend suite (`python -m pytest tests/ -v`), `black --check .` and `ruff check .` in `app/backend` and verify they pass
- [x] 6.3 Run `openspec validate study-upload-endpoint --strict` and verify it passes

## Workflow follow-up

- Open the PR to `develop` with `/create-pull-request` (title `AB-118: ...`, `Closes #118`).
- Archive the change after the PR is merged.
