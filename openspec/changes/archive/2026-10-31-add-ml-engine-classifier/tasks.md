# Tasks

## 1. Dependencies and CI

- [x] 1.1 Install ml_engine in the local backend env (`pip install -r app/ml_engine/requirements.lock` then `pip install --no-deps -e app/ml_engine`); verify that `python -c "from src.pipelines import BundleInferencePipeline, FakeInferencePipeline"` succeeds from `app/backend` (if the Python 3.14 venv cannot install torch, use a 3.11 venv and note it in the PR)
- [x] 1.2 In `.github/workflows/tests.yml`, install the ml_engine lock and package in the backend job before the backend, and add `app/ml_engine/**` to the backend path filter (design D7, Risks); verify the workflow YAML parses and the backend job lists both installs

## 2. Domain errors and HTTP error body

- [x] 2.1 Add optional `details` to `DomainError`, plus `ModelUnavailableError`, `UnsupportedImageError` and `QualityCheckFailedError` in `domain/errors.py` (design D5); verify with unit tests for code, status code and details of each
- [x] 2.2 Make `error_handlers.py` include `details` only when set; verify with `tests/test_error_mapping.py` that a 503 has body `{code, message}` and a 422 quality error has `details.reason`, `snr`, `entropy` and `skull_stripped`

## 3. Port and ml_engine adapter

- [x] 3.1 Change `ClassifierPort` to `async classify(study_id: str, image: bytes) -> Prediction` plus `is_ready()` (design D1), and add `"src"` to `FORBIDDEN_IMPORTS` in `test_architecture_boundaries.py`; verify the architecture test passes and fails if `from src.errors import ...` is temporarily added to `application/`
- [x] 3.2 Add `adapters/outbound/ml/ml_engine_classifier.py` with an injectable loader and `async load(path)` in a worker thread, covering the UNLOADED, READY and UNAVAILABLE states with a logged reason (design D2); verify unit tests for path unset, `ModelUnavailableError` from the loader and an unexpected exception, all leaving `is_ready()` false with the reason logged
- [x] 3.3 Implement `classify` with `anyio.to_thread.run_sync` and `CapacityLimiter(1)`, the `PredictionResult` to `Prediction` mapping and the error mapping (design D3, D4, D5); verify unit tests with `FakeInferencePipeline` (fields, study_id, probabilities summing to 1), doubles raising `UnsupportedImageError` and `QualityCheckError` (details kept), `MODEL_UNAVAILABLE` when not ready, `predict` running on a thread other than the event loop's, and two concurrent calls never overlapping
- [x] 3.4 Delete `adapters/outbound/ml/fake_classifier.py`; verify that `grep -r FakeClassifier app/backend` finds nothing

## 4. Startup wiring and health

- [x] 4.1 In `main.py`, add the `classifier` parameter to `create_app`, default it to `MLEngineClassifier()` and call `await classifier.load(settings.model_bundle_path)` in the lifespan before `yield` (design D2); verify a test where a counting loader is called exactly once across startup and several `classify` calls
- [x] 4.2 Update `tests/test_health.py`: `model_loaded` is true with a fake-loaded classifier and false when the bundle path is unset or the loader raises `ModelUnavailableError`, and `/health` answers while a slow fake inference is running; verify `pytest tests/test_health.py tests/test_api_documentation.py` passes
- [x] 4.3 Add `tests/integration/test_ml_engine_bundle.py`: it skips unless `MODEL_BUNDLE_PATH` is an existing folder, loads the real bundle, and classifies `ML_SAMPLE_IMAGE` only when that file exists. Widen the `integration` marker description in `pyproject.toml`; verify that `pytest tests/integration/test_ml_engine_bundle.py -rs` reports a skip with a clear reason on a machine without a bundle

## 5. Docker and configuration

- [x] 5.1 Update `app/backend/Dockerfile` to copy ml_engine from the `ml_engine` build context into `/opt/ml_engine`, install the lock and `--no-deps` package before the backend (design D6); verify that `docker compose build backend` succeeds and `docker compose run --rm backend python -c "import src.pipelines"` works
- [x] 5.2 Update the `backend` service in `docker-compose.yml`: `additional_contexts`, `MODEL_BUNDLE_PATH: ${MODEL_BUNDLE_PATH-/models/current}` and a read-only bundle volume from `${MODEL_BUNDLE_DIR:-./app/ml_engine/artifacts/current}` (design D6); verify with `docker compose config` that the mount and the variable render
- [x] 5.3 Document `MODEL_BUNDLE_PATH`, `MODEL_BUNDLE_DIR` and the `--build-context ml_engine=app/ml_engine` flag for plain `docker build` in `.env.example` and `app/backend/README.md`; verify the documented commands run as written

## 6. Manual verification in docker-compose (scenarios A and B)

Run `docker compose up --build --no-deps backend postgres` (without `celery_worker`; `--no-deps` because the `minio/minio` image can no longer be pulled) and record the outputs in the PR.

- [x] 6.1 A, no bundle (`MODEL_BUNDLE_PATH= docker compose up ...`): the container stays up, the log says "MODEL_BUNDLE_PATH not set", and `curl localhost:8000/health` returns 200 with `model_loaded:false, database_reachable:true`
- [x] 6.2 B1, missing folder (default path, no `artifacts/current` on the host): the log names the missing `manifest.json` and `/health` reports `model_loaded:false`
- [x] 6.3 B2, corrupt manifest (`MODEL_BUNDLE_DIR` pointing to a folder whose `manifest.json` contains `{garbage`): the log says "invalid artifact manifest" and `/health` reports `model_loaded:false`
- [x] 6.4 B3, incomplete bundle (a valid-looking `manifest.json` without `classifier.joblib`): the log names the missing file or the sha256 mismatch and `/health` reports `model_loaded:false`
- [x] 6.5 In every scenario, `docker compose logs backend` shows exactly one `model unavailable` line after several `/health` calls, `/docs` loads, and `docker stats` RAM at idle is noted as the baseline for AB-120
- [x] 6.6 The full backend suite passes in the container: `docker compose run --rm backend pytest tests/ -rs`, with the ml bundle integration test skipped and its reason shown

## 7. Scenario C with a trained bundle

The dataset is on `develop`. The bundle was trained locally (`python -m src.training ... --version 0.1.0`, test macro-F1 0.804), and `artifacts/` is git-ignored.

- [x] 7.1 Keep the dataset (`app/ml_engine/src/data`, 178 MB) out of the backend image by installing ml_engine through a `RUN --mount=type=bind` instead of `COPY` (design D6); verify the image still imports `src.pipelines` and contains no `/tmp/ml_engine` or dataset image
- [x] 7.2 C, valid bundle (`MODEL_BUNDLE_DIR=./app/ml_engine/artifacts/model-v0.1.0`): `/health` returns `model_loaded:true` and a Test JPG gives a prediction with model version `0.1.0`. A text file gives 422 `UNSUPPORTED_IMAGE`, and a flat PNG gives 422 `QUALITY_CHECK_FAILED` with snr, entropy and skull_stripped
- [x] 7.3 The integration test runs against the real bundle inside the container (`ML_SAMPLE_IMAGE` set) and the full suite passes with nothing skipped

## Workflow follow-up

- The 503 and 422 answers of a real HTTP upload are verified in AB-118.
- Choose the Cloud Run build context and bundle location in AB-120.
- Archive the change after the PR to `develop` is merged.
