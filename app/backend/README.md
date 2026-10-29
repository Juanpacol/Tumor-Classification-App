# Ether Backend

FastAPI backend, hexagonal architecture (see root `CLAUDE.md`).

## Muse Spark / Meta Model API

Verified against Meta's official docs (`dev.meta.ai/docs`) on 2026-10-28, implemented in
`app/adapters/outbound/llm/meta_model_llm.py`.

- **Base URL**: `https://api.meta.ai/v1`
- **Chat completions endpoint**: `POST /chat/completions`
- **Model ID**: `muse-spark-1.3` (latest; also available: `muse-spark-1.2`, `muse-spark-1.1`,
  and `-contributor` variants that trade a lower price for Meta training on your data — not used)
- **Auth**: `Authorization: Bearer $META_MODEL_API_KEY`
- **Streaming**: `"stream": true` returns Server-Sent Events; each chunk is
  `choices[0].delta.content`, terminated by a `data: [DONE]` line.
- **Structured JSON output**: `response_format: {"type": "json_schema", "json_schema": {"name",
  "schema", "strict": true}}`. With `strict: true` every object in the schema needs
  `additionalProperties: false` and a complete `required` list.
- **Images**: sent inline as `image_url: {"url": "data:image/png;base64,<...>"}` on the last
  message. Supported formats: jpeg/png/gif/webp/ico. Max 50 MB per inline image, 50 images per
  request. The adapter only ever sends PNG, longest side ≤1024 px (rendered by the caller from
  the stored de-identified file, never the raw DICOM).
- **Rate limits**: Meta has not published authoritative numeric limits for the preview; the
  adapter does not assume a specific number and instead retries once with a fixed backoff on
  HTTP 429 or 5xx (and on a connect/read timeout), with no retry on 401/403.

### Error mapping

`MetaModelLLM` raises `DomainError` (`app/domain/errors.py`), which the FastAPI handler in
`app/adapters/inbound/http/error_handlers.py` turns into `{code, message}`:

| Condition | code | HTTP status |
|---|---|---|
| `META_MODEL_API_KEY` not set | `LLM_UNAVAILABLE` | 503 |
| 401/403 from Meta | `LLM_AUTH_FAILED` | 502 |
| Other 4xx/5xx after the retry | `LLM_REQUEST_FAILED` | 502 |
| Timeout after the retry | `LLM_TIMEOUT` | 504 |

### Token usage

Usage (`usage` field of the response) is logged per call via the standard `logging` module
under the `app.adapters.outbound.llm.meta_model` logger, to track spend against the hackathon's
USD 150 credit budget. The API key itself is never logged.

## Choosing the LLM

`LLM_PROVIDER` picks the adapter when the app starts (`adapters/outbound/llm/factory.py`):

- `meta` (the default): the real Muse Spark adapter. **Without `META_MODEL_API_KEY` the app still starts and every LLM call answers 503 `LLM_UNAVAILABLE`.** It never falls back to invented answers, so a deployment cannot serve fake reports.
- `fake`: canned answers, only for local development without a key. Its report says it is a demo and is uncertain by construction.

The startup log says which one is in use and whether a key is set, never the key. The HTTP client is closed when the app shuts down, and the adapter is available to routers as `app.state.llm`.

## Spend cap

Meta offers no hard spending cap (only an e-mail alert) and the backend is public, so the backend enforces its own, behind the real LLM adapter (`adapters/outbound/llm/spend_cap.py`):

- `LLM_DAILY_TOKEN_BUDGET` (default 1,000,000 tokens a day, UTC) and `LLM_MAX_REQUESTS_PER_MINUTE` (default 30). When either is reached the call answers **429**, `LLM_BUDGET_EXCEEDED` or `LLM_RATE_LIMITED`, and never reaches Meta. Both limits are logged at startup and when they are hit.
- The tokens come from the `usage` Meta returns. A call that reports none counts a default of 4,000. Streamed answers carry no usage, so they are **estimated** (4 characters a token, plus 1,500 for an image), and what was sent counts even if the reader stops early. These two numbers are guesses until the live validation (AB-161) measures them.
- The limits are checked before a call and the tokens are added after it, so one call can overshoot. A stream is checked when iteration starts, not when it is created.
- **Limits of this design:** the counters live in memory. They reset when the process restarts (Cloud Run scales to zero) and each instance counts on its own, so with N instances the real cap is up to N times larger. A counter in the database would remove that, and has not been built.
- The fake LLM is not capped: it costs nothing.

## Clinical report (core of AB-122)

`application/` holds the part of the report that needs no HTTP and no database:

- `report_prompt.py`: the system prompt, versioned as `PROMPT_VERSION` (`report-v1`), and the user message with the classifier output. The clinical notes are placed between `<notes>` markers as untrusted data, and the closing marker is stripped from them so they cannot break out.
- `clinical_report.py`: `ClinicalReportOutput`, the Pydantic schema the model must fill (`extra="forbid"`, `ai_agreement` is `agrees`, `disagrees` or `uncertain`), and the disclaimer, which the backend adds and the model never writes.
- `report_guardrails.py`: rejects definitive wording ("the patient has", "diagnosed with", ...), treatment or medication wording, and, when triage is flagged or the model does not agree, a report that does not state the uncertainty and recommend expert review.
- `generate_report.py`: `GenerateClinicalReport`. Only COMPLETED studies (404 for an unknown one, 409 for the rest), returns the stored report unless `regenerate` is set or the last one failed, and retries once with the reasons when the output is invalid. If it is still invalid the failure is stored and the answer is 502.

Still missing: the `POST /api/studies/{id}/report` endpoint (it needs the study upload of AB-118), the check with the live API (AB-161) and the manual review of 10 reports from the public Test images.

## Data sent to Meta while free credits are in use

The Billing page of the Meta console says: "While using free credits your content may be used for product improvement" (read on 2026-10-29). So, while the free credits are in use:

- Send only images of the public dataset and invented notes. Never real patient data.
- Do not use the `-contributor` model variants either: they are cheaper in exchange for training on the content.
- Before any use with real data, either stop using the free credits (the "Stop using free credits" link in Billing, which moves the usage to the card) or do not send that data to this service.


## MRI classifier (ml_engine)

`adapters/outbound/ml/ml_engine_classifier.py` puts the ml_engine `InferencePipeline` behind `ClassifierPort`. It is the only module that imports ml_engine (`src.*`), and `tests/test_architecture_boundaries.py` keeps it out of `domain/` and `application/`.

- **Install** (local): `pip install -r ../ml_engine/requirements.lock`, then `pip install --no-deps -e ../ml_engine`, then the backend as usual.
- **Model bundle**: `MODEL_BUNDLE_PATH` is the folder the ml_engine training writes (`artifacts/model-v<version>/`). It is loaded **once, at startup** (lifespan), in a worker thread.
- **No usable model is a normal state.** If `MODEL_BUNDLE_PATH` is unset, the folder is missing, or the bundle is incomplete or altered, the app still starts, logs `model unavailable: <reason>`, `GET /health` reports `model_loaded: false`, and every classification answers 503 `MODEL_UNAVAILABLE`.
- **Inference** runs in a worker thread, one at a time per process, so the event loop keeps serving other requests.

| Condition | code | HTTP status | `details` |
|---|---|---|---|
| No model loaded | `MODEL_UNAVAILABLE` | 503 | — |
| File is not a supported image | `UNSUPPORTED_IMAGE` | 422 | `reason` |
| Image fails the quality gate | `QUALITY_CHECK_FAILED` | 422 | `reason`, `snr`, `entropy`, `skull_stripped` |

### Docker

The image installs ml_engine from a second build context named `ml_engine`. docker-compose passes it. A plain build needs it too:

```bash
docker build --build-context ml_engine=app/ml_engine -t ether-backend app/backend
```

In docker-compose the bundle is mounted read-only at `/models/current` from `MODEL_BUNDLE_DIR` (default `./app/ml_engine/artifacts/current`):

```bash
docker compose up --build --no-deps backend postgres              # bundle from the default folder
MODEL_BUNDLE_DIR=./app/ml_engine/artifacts/model-v0.1.0 docker compose up --no-deps backend postgres
MODEL_BUNDLE_PATH= docker compose up --no-deps backend postgres   # start without a model
```

`--no-deps` skips Redis and MinIO, which the backend does not use yet (the `minio/minio` image can no longer be pulled from Docker Hub). If the default folder does not exist, Docker creates it empty and the app starts without a model.

The integration test `tests/integration/test_ml_engine_bundle.py` runs only when `MODEL_BUNDLE_PATH` points to a bundle folder, and classifies `ML_SAMPLE_IMAGE` when that file exists.
