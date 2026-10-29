# Design

## Context

- `ClassifierPort` is synchronous and takes a path (`classify(image_path) -> Prediction`). `main.create_app` builds a `FakeClassifier` eagerly and passes it to `create_health_router`.
- ml_engine exposes `InferencePipeline.predict(file_bytes) -> PredictionResult`, a frozen dataclass with `TumorLabel` enum keys and nested `QualityMetrics` / `FeatureContribution`. `BundleInferencePipeline.from_bundle(dir)` verifies every file against its manifest and raises `ModelUnavailableError(reason)` when anything is wrong. `predict` raises `UnsupportedImageError(reason)` or `QualityCheckError(reason, metrics)`.
- The ml_engine package is imported as `src` (decisions.md D2). It is installed with `pip install -r app/ml_engine/requirements.lock` followed by `pip install --no-deps app/ml_engine` (AB-159). Importing `src.pipelines` pulls in torch, even for `FakeInferencePipeline`.
- `DomainError(code, message, status_code)` is serialized as `{code, message}` by `register_exception_handlers`.
- The backend Docker build context is `./app/backend`, so it cannot see `app/ml_engine`. The backend CI job installs only the backend package.
- Constraint: the hexagonal layout stays flat. New code goes into the existing `domain/`, `ports/` and `adapters/` folders, with no new top-level folders in `app/backend/`.

## Goals / Non-Goals

**Goals:**
- `domain/` and `application/` never import ml_engine. Only `adapters/outbound/ml/` does.
- The app can be tested end to end in docker-compose without a trained bundle (scenarios A and B).

**Non-Goals:**
- Upload endpoint, classification use case and study status transitions (AB-118).
- Verification with a trained bundle and real images (scenario C) until the dataset is in the repo.
- Reducing the image size (dev tools in the lock, multi-stage build) and the Cloud Run build setup (AB-120).
- Fixing the `celery_worker` compose service.

## Decisions

### D1. Port: `async classify(study_id: str, image: bytes) -> Prediction`
The port returns the existing domain `Prediction`, and the adapter copies `study_id` into it. Being async lets the adapter decide how to offload CPU work, so the application layer never knows the work is CPU-bound, the same choice made for `StudyRepositoryPort` in AB-116. `is_ready() -> bool` stays synchronous because it only reads state.
- *Alternative:* a separate `Classification` value object without `study_id`, assembled into `Prediction` by the use case. Rejected because it duplicates seven fields for no behavior.

### D2. One adapter with a load step, wired in the lifespan
`MLEngineClassifier` lives in `adapters/outbound/ml/ml_engine_classifier.py` and has three states:

```
 UNLOADED --load(path)--> READY(pipeline)
     |                     
     +--------------------> UNAVAILABLE(reason)
```

- Its constructor takes `loader: Callable[[str], InferencePipeline]`, with `BundleInferencePipeline.from_bundle` as the default. Tests pass a loader that returns `FakeInferencePipeline` and counts its calls.
- `async load(path: str | None)` runs the loader in a worker thread, because loading torch weights blocks. If `path` is None, the state becomes `UNAVAILABLE("MODEL_BUNDLE_PATH not set")`. `ModelUnavailableError` becomes `UNAVAILABLE(exc.reason)` with a WARNING log. Any other exception is logged with its traceback and also leaves the adapter `UNAVAILABLE`, so a bad bundle never stops startup.
- `create_app(..., classifier: ClassifierPort | None = None)`: if no classifier is given, it builds `MLEngineClassifier()`. The lifespan calls `await classifier.load(settings.model_bundle_path)` before `yield` only when the classifier has a `load` method (an injected test double can be ready already). The health router keeps receiving the same instance, so it sees the state change without rebuilding the router.
- *Alternative:* load inside `create_app`. Rejected because the acceptance criteria require the lifespan, and blocking at import time would slow `TestClient` construction and uvicorn reload.

### D3. Inference offloading: `anyio.to_thread.run_sync` with a `CapacityLimiter(1)`
- `predict` is CPU-bound and torch already uses several threads internally. More than one inference at a time on a 2–4 GB Cloud Run instance risks OOM and gains no throughput, so the limiter keeps inferences serialized while the event loop stays free.
- anyio is already installed with FastAPI and Starlette uses it for its own thread offloading.
- *Alternative:* `asyncio.to_thread` with the default executor. Rejected because it does not bound concurrency.

### D4. Mapping `PredictionResult` to `Prediction`
`label = result.label.value`, `probabilities = {k.value: v}`, `quality = asdict(result.quality)`, `top_features = [asdict(f) for f in result.top_features]`, `model_version` is copied. `Prediction` already stores dicts and lists, so persistence (AB-116) needs no change.

### D5. Domain errors carry `details`
- `DomainError` gains an optional `details: dict[str, Any] | None`. Subclasses in `domain/errors.py`:
  - `ModelUnavailableError(reason)` → `MODEL_UNAVAILABLE`, 503
  - `UnsupportedImageError(reason)` → `UNSUPPORTED_IMAGE`, 422, `details={"reason"}`
  - `QualityCheckFailedError(reason, metrics: dict)` → `QUALITY_CHECK_FAILED`, 422, `details={"reason", **metrics}`
- The domain cannot import ml_engine types, so the adapter turns `QualityMetrics` into a dict. In the adapter, ml_engine errors are imported through their module (`from src import errors as ml_errors`) to avoid name clashes.
- The handler adds `details` to the body only when it is set, so existing error bodies do not change.
- An `ErrorResponse` schema and the per-endpoint `responses={503: ...}` documentation are added with the first endpoint that raises these errors (AB-118). `/health` never raises them.

### D6. Docker: `additional_contexts` instead of moving the build context
- In compose, the backend `build` gains `additional_contexts: { ml_engine: ./app/ml_engine }`. The Dockerfile copies `requirements.lock`, `pyproject.toml` and `src/` from that context into `/opt/ml_engine`, installs the lock, then runs `pip install --no-deps /opt/ml_engine` (not editable). Both happen before the backend install, so the lock pins win.
- ml_engine lives outside `/app`, so the `./app/backend:/app` dev mount does not hide it.
- Bundle: `MODEL_BUNDLE_PATH: ${MODEL_BUNDLE_PATH-/models/current}` and the volume `${MODEL_BUNDLE_DIR:-./app/ml_engine/artifacts/current}:/models/current:ro`. Setting `MODEL_BUNDLE_PATH=` (empty) gives scenario A, because settings ignore empty values and it becomes unset. A host folder that does not exist gives B1, because Docker mounts an empty folder.
- *Alternative:* build from the repo root. That also works for plain `docker build` and Cloud Build, but every `COPY` changes and the whole repo becomes build context. That choice is deferred to AB-120 (see Risks).

### D7. Tests
- Unit tests (`tests/test_ml_engine_classifier.py`, `tests/test_health.py`, `tests/test_error_mapping.py`) use `FakeInferencePipeline` and small pipeline doubles that raise ml_engine errors or sleep. They check: mapping, both error mappings with details, the three UNAVAILABLE causes, one load across N classifications, `predict` running on a thread other than the event loop's, and `/health` answering while a slow fake inference runs.
- `test_architecture_boundaries.py` adds `"src"` to `FORBIDDEN_IMPORTS`.
- Integration (`tests/integration/test_ml_engine_bundle.py`): it is skipped unless `MODEL_BUNDLE_PATH` points to an existing folder, and it classifies `ML_SAMPLE_IMAGE` only when that file exists. It uses no DB fixtures, so the Postgres skip does not affect it. The `integration` marker description is widened to "external resource (PostgreSQL or model bundle)".
- CI (`tests.yml`, backend job) installs `app/ml_engine/requirements.lock` and `--no-deps app/ml_engine` before the backend.

## Risks / Trade-offs

- [Plain `docker build app/backend` and Cloud Build lack the `ml_engine` context] → document `--build-context ml_engine=app/ml_engine`, and decide between that and a root context in AB-120.
- [Image size and CI time grow with torch CPU (~800 MB installed) and the dev tools in the lock] → pip cache in CI. Producing a runtime-only lock is follow-up work for the ML owner.
- [Local venv is Python 3.14 while the lock targets 3.11] → docker-compose (3.11) is the reference environment for this story's manual checks.
- [The CI path filter does not run backend tests when only `app/ml_engine` changes, so a contract change could break the adapter unnoticed] → add `app/ml_engine/**` to the backend filter.
- [`src` is a generic top-level name (D2 in decisions.md)] → the architecture test guards the pure layers, and a rename only touches the adapter.
- [The quality gate thresholds are provisional (decisions.md D7), so real uploads may be rejected with 422] → the response's `details` carries the metrics, which helps calibration once the dataset arrives.

## Migration Plan

1. Merge with no bundle mounted anywhere: production behavior equals scenario A (`model_loaded: false`). No endpoint uses the classifier yet, so nothing user-facing changes.
2. Once a trained bundle exists, mount it at `MODEL_BUNDLE_PATH` (compose volume locally, Cloud Run volume or baked into the image in AB-120).
3. Rollback: revert the commit. There are no schema or data changes.

## Open Questions

- Where the bundle lives in Cloud Run (baked into the image, GCS volume mount, or downloaded at startup). Resolve in AB-120. It does not affect the adapter, which only reads a folder path.
