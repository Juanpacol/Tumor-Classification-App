# Proposal

## Why

The backend still classifies with a hard-coded `FakeClassifier`, and `ClassifierPort.classify(image_path)` does not match the ml_engine contract, `InferencePipeline.predict(file_bytes)` (AB-117). The upload flow (AB-118) needs a real classifier behind the port. That classifier has to load the model bundle once, keep the app running when the bundle is missing, and never block the event loop with CPU-bound inference.

## What Changes

- **BREAKING** `ClassifierPort.classify` becomes `async classify(study_id: str, image: bytes) -> Prediction`. `is_ready()` stays.
- Add an ml_engine adapter in `adapters/outbound/ml/` that wraps `BundleInferencePipeline` (or any `InferencePipeline`). It loads the bundle once in the FastAPI lifespan, runs `predict` in a worker thread with at most one inference at a time, and maps `PredictionResult` to the domain `Prediction`.
- A missing or invalid bundle is a normal state. The app starts and logs the reason, `/health` reports `model_loaded: false`, and `classify` raises a domain `MODEL_UNAVAILABLE` error that maps to HTTP 503.
- ml_engine `UnsupportedImageError` and `QualityCheckError` become domain errors (`UNSUPPORTED_IMAGE` and `QUALITY_CHECK_FAILED`, HTTP 422) that keep the `reason` and the quality metrics. The HTTP error body gains an optional `details` object.
- Remove the backend's own `FakeClassifier`. Tests use ml_engine's `FakeInferencePipeline` instead.
- The architecture test forbids importing `src` (the ml_engine package name) from `domain/` and `application/`.
- The backend Docker image installs ml_engine from `requirements.lock` (CPU-only PyTorch). docker-compose passes `app/ml_engine` as an additional build context and reads the bundle from a read-only mount at `MODEL_BUNDLE_PATH`.
- The backend CI job installs ml_engine so tests can import the fake pipeline.

## Capabilities

### New Capabilities
- `mri-classification`: classifying an MRI image through the ml_engine, including model availability, its health reporting and the errors returned for unusable images.

### Modified Capabilities
<!-- None: no specs exist yet in openspec/specs/. -->

## Impact

- **Code**: `ports/classifier_port.py`, `domain/errors.py`, `adapters/outbound/ml/` (new `ml_engine_classifier.py`; `fake_classifier.py` deleted), `adapters/inbound/http/error_handlers.py`, `main.py` (lifespan loading, `classifier` injection in `create_app`), `.env.example`. No new folders in `app/backend/`.
- **Tests**: unit tests with `FakeInferencePipeline`, and one integration test in `tests/integration/` with the real bundle, skipped when the bundle is absent. `test_health.py` and `test_architecture_boundaries.py` are updated.
- **Dependencies**: the backend now depends on the ml_engine package (`torch` CPU, `scikit-learn`, `numpy`...). The image gets bigger and the CI install takes longer.
- **Systems**: `app/backend/Dockerfile`, `docker-compose.yml` (backend service), `.github/workflows/tests.yml`. These also affect the Cloud Run build (AB-120).
- **Out of scope**: the upload endpoint and the classification use case (AB-118), so the HTTP 503 on upload is verified in AB-118. Also out of scope: verification with a trained bundle and real images until the dataset is in the repository, and the broken `celery_worker` compose service.
