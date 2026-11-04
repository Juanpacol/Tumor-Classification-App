# Ether Backend

FastAPI backend, hexagonal architecture (see root `CLAUDE.md`).

## Muse Spark / Meta Model API

Written against Meta's docs (`dev.meta.ai/docs`, 2026-10-28) and **checked with live calls on 2026-11-03** (AB-161, see *Live validation*). Implemented in `adapters/outbound/llm/meta_model_llm.py`.

- **Base URL**: `https://api.meta.ai/v1`
- **Chat completions endpoint**: `POST /chat/completions`. `/v1/responses` also exists and is not used.
- **Model ID**: `muse-spark-1.3`. `GET /v1/models` also lists `muse-spark-1.2` and `muse-spark-1.1`. It also lists `-contributor` variants, which are cheaper because Meta trains on the content; those are not used.
- **Auth**: `Authorization: Bearer $META_MODEL_API_KEY`
- **Token bound**: every request sends `max_completion_tokens` (`META_MODEL_MAX_COMPLETION_TOKENS`, default 4096). It counts reasoning *and* visible tokens, and `max_tokens` is deprecated.
- **Reasoning effort**: the report (`complete_json`) sends `META_MODEL_REPORT_REASONING_EFFORT` (default `low`). The chat (`stream`) sends `META_MODEL_CHAT_REASONING_EFFORT` (default `minimal`). Accepted values are `minimal`, `low`, `medium`, `high`, `xhigh` and `max`; `muse-spark-1.3` answers 400 to `none`. The two differ because a stream sends nothing while the model reasons (see the table).
- **Streaming**: `"stream": true` plus `stream_options: {"include_usage": true}` returns Server-Sent Events. Each `data: {...}` line is followed by a blank line, and the text is in `choices[0].delta.content`. Live, `usage` arrives on the **final content chunk** (the one with `finish_reason: "stop"`), followed by `data: [DONE]`. Meta's schema says it can instead come on a chunk with `choices: []`; the adapter accepts both.
- **Structured JSON output**: `response_format: {"type": "json_schema", "json_schema": {"name", "schema", "strict": true}}`. With `strict: true` every object needs `additionalProperties: false` and a complete `required` list. Pydantic's `$defs`/`$ref` were accepted live as sent.
- **Images**: sent inline on the last message as `image_url: {"url": "data:<type>;base64,<...>"}`. The type is detected from the bytes: `image/jpeg` or `image/png`, which are the only formats an upload can have. The image is the stored upload itself, without metadata and not re-encoded or resized. Meta also accepted JPEG data labelled `image/png`, but the adapter no longer relies on that. A 354x442 dataset image cost about 184 prompt tokens.
- **Rate limits** (pricing page, 2026-11-03): 3,000 requests and 4,000,000 tokens a minute, **per team, not per key**. The adapter retries once with a fixed backoff on HTTP 429 or 5xx, and on a connect/read timeout. It never retries on 401/403.

### Error mapping

`MetaModelLLM` raises `DomainError` (`domain/errors.py`), which the FastAPI handler in `adapters/inbound/http/error_handlers.py` turns into `{code, message}`:

| Condition | code | HTTP status |
|---|---|---|
| `META_MODEL_API_KEY` not set | `LLM_UNAVAILABLE` | 503 |
| 401/403 from Meta | `LLM_AUTH_FAILED` | 502 |
| Other 4xx/5xx after the retry | `LLM_REQUEST_FAILED` | 502 |
| Timeout after the retry | `LLM_TIMEOUT` | 504 |

An image that is neither JPEG nor PNG raises `ValueError` before anything is sent. It cannot happen through the upload endpoint.

### Token usage

The `usage` of every call is logged under the `adapters.outbound.llm.meta_model` logger and handed to the spend cap. A stream reports it exactly once: Meta's own figure, or an estimate when the stream ended without one (see *Spend cap*). `completion_tokens` includes `completion_tokens_details.reasoning_tokens`. The API key is never logged.

### Live validation (2026-11-03)

Run with `scripts/probe_muse_spark.py`, from `app/backend` with the key in `.env`:
- `raw` builds each request by hand, so Meta's own error body is visible;
- `adapter` goes through `build_llm(Settings())`.

The inputs were the public dataset image `ml_engine/src/data/Test/glioma/Te-gl_1.jpg` and an invented case: notes, classifier output and question. The probe refuses `-contributor` models and never prints the key. The test suite never calls Meta; `tests/fixtures/meta_model/` holds a report response and a stream captured in this run.

Prices from Meta's pricing page, read 2026-11-03, `muse-spark-1.3` standard mode, per 1M tokens: **USD 1.25 input, 0.15 cached input, 4.25 output**. Cost = (prompt - cached) x input + cached x cached input + completion x output.

| Capability | Path | Effort | HTTP | Latency (s) | First token (s) | Prompt | Cached | Completion | Reasoning | USD |
|---|---|---|---|---|---|---|---|---|---|---|
| Text | raw | not set | 200 | 12.02 | - | 20 | 0 | 924 | 866 | 0.00395 |
| Image (JPEG, `image/jpeg`) | raw | not set | 200 | 24.55 | - | 204 | 99 | 1,991 | 1,940 | 0.00861 |
| Image (JPEG labelled `image/png`) | raw | not set | 200 | 20.07 | - | 204 | 0 | 1,621 | 1,558 | 0.00714 |
| Structured JSON (report + image) | raw | `low` | 200 | 16.39 | - | 1,282 | 0 | 1,659 | 1,139 | 0.00865 |
| Structured JSON (report + image) | raw | `medium` | 200 | 21.70 | - | 1,282 | 0 | 2,165 | 1,648 | 0.01080 |
| Streaming (chat + image) | raw | `none` | 400 | 0.14 | - | - | - | - | - | 0 |
| Streaming (chat + image) | raw | `minimal` | 200 | 3.63 | 2.69 | 764 | 0 | 242 | 163 | 0.00198 |
| Streaming (chat + image) | raw | `low` | 200 | 8.39 | 7.84 | 764 | 0 | 744 | 679 | 0.00412 |
| Structured JSON (report + image) | adapter | `low` | 200 | 15.57 | - | 1,282 | 0 | 1,485 | 865 | 0.00791 |
| Streaming (chat + image) | adapter | `minimal` | 200 | 3.78 | 3.07 | 764 | 0 | 224 | 143 | 0.00191 |

- Both reports matched `GroundedClinicalReportOutput`, at `low` and at `medium`. `medium` cost about 25 % more and took about 5 s longer, with no gain the schema check can see.
- An earlier run, before the probe used the real chat prompt, streamed with no effort set: 4,231 reasoning tokens and **46.6 s to the first token**. That is why the chat runs at `minimal`.
- The whole validation, two raw runs plus the adapter run, cost about USD 0.11 of the free credits.
- **One run on one day, not a benchmark.** Latency varied by up to 4 s between the two raw runs.
- **Unconfirmed**: that reasoning tokens bill at the output price. The pricing page does not say, and the cost column assumes they do. Check the Billing page after a run to confirm.

Adapter fixes from this validation:
- the real image type;
- `max_completion_tokens` and a reasoning effort for each of report and chat;
- `stream_options.include_usage`, with usage read from whichever chunk carries it;
- measured stream usage reaches the spend cap.

## Choosing the LLM

`LLM_PROVIDER` picks the adapter when the app starts (`adapters/outbound/llm/factory.py`):

- `meta` (the default): the real Muse Spark adapter. **Without `META_MODEL_API_KEY` the app still starts and every LLM call answers 503 `LLM_UNAVAILABLE`.** It never falls back to invented answers, so a deployment cannot serve fake reports.
- `fake`: canned answers, only for local development without a key. Its report says it is a demo and is uncertain by construction.

The startup log says which one is in use and whether a key is set, never the key. The HTTP client is closed when the app shuts down, and the adapter is available to routers as `app.state.llm`.

## Spend cap

Meta offers no hard spending cap (only an e-mail alert) and the backend is public, so the backend enforces its own, behind the real LLM adapter (`adapters/outbound/llm/spend_cap.py`):

- `LLM_DAILY_TOKEN_BUDGET` (default 1,000,000 tokens a day, UTC) and `LLM_MAX_REQUESTS_PER_MINUTE` (default 30). When either is reached the call answers **429**, `LLM_BUDGET_EXCEEDED` or `LLM_RATE_LIMITED`, and never reaches Meta. Both limits are logged at startup and when they are hit.
- The tokens come from the `usage` Meta returns, streams included. A non-streamed call that reports none counts a default of 4,000. A stream that ends without usage, because the reader stopped early or the connection dropped, is **estimated** by the adapter: 4 characters a token, plus 200 for an image (184 measured on 2026-11-03). What was sent counts even if the reader stops early. Each stream is counted once.
- The limits are checked before a call and the tokens are added after it, so one call can overshoot. A stream is checked when iteration starts, not when it is created.
- **Limits of this design:** the counters live in memory. They reset when the process restarts (Cloud Run scales to zero) and each instance counts on its own, so with N instances the real cap is up to N times larger. A counter in the database would remove that, and has not been built.
- The fake LLM is not capped: it costs nothing.

## API

Everything is under `/api/studies`; interactive docs at `/docs`.

| Endpoint | What it does |
|---|---|
| `GET /` | List studies, newest first (`page`, `page_size` up to 100) |
| `POST /` | Upload a study (see *Study upload*) |
| `GET /{id}` | Study with its prediction and report, if any |
| `GET /{id}/image` | The stored, metadata-free image |
| `POST /{id}/report` | Generate the clinical report, or return the stored one |
| `GET /{id}/chat` | The study's conversation, oldest first |
| `POST /{id}/chat` | Ask a question (1 to 2000 characters); the answer streams as Server-Sent Events |

## Clinical report

In `application/`, with no HTTP or database:

- `report_prompt.py`: system prompt (`PROMPT_VERSION = "report-v2"`) and user message. Clinical notes go between `<notes>` markers as untrusted data.
- `grounded_report.py`: the schema the model must fill. Findings, visual observations, impression and differentials are claims, each a `text` plus the `evidence_keys` it rests on. The backend adds the disclaimer; the model never writes it.
- `evidence_bundle.py`: what a claim can cite: `prediction.label`, `.confidence`, `.needs_expert_triage`, `.probabilities.<class>` and `.top_features.<name>`.
- `citation_verifier.py`: a claim is verified only if it cites at least one key, every key exists, every number it states matches a cited value (a probability may be written as a percentage), a claim citing the label names it, and it names another class only when it cites that class's probability. The check is textual, not semantic. Remarks about the image carry no keys and stay unverified by design.
- `report_guardrails.py`: rejects definitive or treatment wording and, when triage is flagged or the model does not agree, a report that does not state the uncertainty.
- `generate_report.py`: only COMPLETED studies (404 unknown, 409 otherwise). Returns the stored report unless the last one failed, and retries once with the reasons if the output is invalid; still invalid means a stored failure and 502.

## Chat

`application/study_chat.py` answers with the classifier output, the evidence bundle, the stored report, the earlier messages and the image. SSE events: `token` (`text`), then `done` (the stored message), or `error` (`code`, `message`). The exchange is stored only when the answer completes. 404 unknown study, 409 without prediction or image.

Live check with the Meta API (2026-11-02, public images, invented classifier output): reports for the four classes, including a wrong prediction, had no guardrail violations and every claim that cited evidence was verified. The token bound, the reasoning efforts and measured usage were added by AB-161 (see *Live validation*). Still missing: a manual review of 10 reports.

## Data sent to Meta while free credits are in use

The Billing page of the Meta console says: "While using free credits your content may be used for product improvement" (read on 2026-10-29). So, while the free credits are in use:

- Send only images of the public dataset and invented notes. Never real patient data.
- Do not use the `-contributor` model variants either: they are cheaper in exchange for training on the content.
- Before any use with real data, either stop using the free credits (the "Stop using free credits" link in Billing, which moves the usage to the card) or do not send that data to this service.


## Study upload (AB-118)

`POST /api/studies` takes a multipart `file` and optional `clinical_notes` (at most 2000 characters) and classifies the image in the same request (`application/upload_study.py`, `application/image_upload.py`).

- **Checked before anything is stored**, in this order: a model is loaded, the file is at most 20 MB, its content (never its name or content type) is a JPEG or PNG, and its pixels decode fully. DICOM, other formats and corrupt files are 415.
- **Stored without metadata**: EXIF, XMP, IPTC, comments and PNG text chunks are removed without re-encoding, so the stored and classified pixels are the uploaded ones. The key is `studies/<study uuid>`; the original filename is not kept anywhere. Removing EXIF also removes its orientation tag, so an image is classified as stored, not rotated.
- **No PHI in the logs**: an unexpected classifier error is logged as the study id, the error type and the raising file and line, never its message, the notes or the filename.

| Outcome | Stored | HTTP status | Body |
|---|---|---|---|
| Classified | `completed` + prediction | 201 | the study |
| Classifier rejects the image | `rejected` + reason | 422 | `code` (`QUALITY_CHECK_FAILED` or `UNSUPPORTED_IMAGE`), `message`, `details`, `study` |
| Unexpected classifier error | `failed` | 500 | `INTERNAL_ERROR`, generic message |
| Missing file or notes too long | nothing | 422 | FastAPI validation `detail` |
| Over 20 MB | nothing | 413 | `FILE_TOO_LARGE` |
| Not a decodable JPEG/PNG | nothing | 415 | `UNSUPPORTED_MEDIA_TYPE` |
| No model loaded | nothing | 503 | `MODEL_UNAVAILABLE` |

Any other unhandled error in the API answers 500 with a generic `INTERNAL_ERROR` body.

## MRI classifier (ml_engine)

`adapters/outbound/ml/ml_engine_classifier.py` puts the ml_engine `InferencePipeline` behind `ClassifierPort`. It is the only module that imports ml_engine (`src.*`), and `tests/test_architecture_boundaries.py` keeps it out of `domain/` and `application/`.

- **Install** (local): `pip install -r ../ml_engine/requirements.lock`, then `pip install --no-deps -e ../ml_engine`, then the backend as usual.
- **Model bundle**: `MODEL_BUNDLE_PATH` is the folder the ml_engine training writes (`artifacts/model-v<version>/`). It is loaded **once, at startup** (lifespan), in a worker thread.
- **No usable model is a normal state.** If `MODEL_BUNDLE_PATH` is unset, the folder is missing, or the bundle is incomplete or altered, the app still starts, logs `model unavailable: <reason>`, `GET /health` reports `model_loaded: false`, and every classification answers 503 `MODEL_UNAVAILABLE`.
- **Inference** runs in a worker thread, one at a time per process, so the event loop keeps serving other requests.

| Condition | code | HTTP status | `details` |
|---|---|---|---|
| No model loaded | `MODEL_UNAVAILABLE` | 503 | — |
| Decodable image the model cannot use (e.g. under 64 px) | `UNSUPPORTED_IMAGE` | 422 | `reason` |
| Image fails the quality gate | `QUALITY_CHECK_FAILED` | 422 | `reason`, `snr`, `entropy`, `skull_stripped` |

### Docker

The image is built with the repository root as context, so it can install ml_engine from the sibling folder (docker-compose does it). A plain build:

```bash
docker build -f app/backend/Dockerfile -t ether-backend .
```

The image carries the deployed model: the `runtime` stage downloads the GitHub Release `model-v<MODEL_VERSION>` (build argument, default `0.1.0`) with `python -m src.release fetch`, which checks its SHA-256, and sets `MODEL_BUNDLE_PATH` to it. Cloud Run needs no extra setting, and the build fails if the release cannot be downloaded or verified. To deploy another version, publish its release (`app/ml_engine/README.md`, *Publish a model*) and change the default of `MODEL_VERSION`.

In docker-compose a bundle is mounted read-only at `/models/current` from `MODEL_BUNDLE_DIR` (default `./app/ml_engine/artifacts/current`):

```bash
docker compose up -d --build backend postgres                      # bundle from the default folder
MODEL_BUNDLE_DIR=./app/ml_engine/artifacts/model-v0.1.0 docker compose up -d backend postgres
MODEL_BUNDLE_PATH= docker compose up -d backend postgres           # start without a model
```

If the default folder does not exist, Docker creates it empty and the app starts without a model. The full start-up guide is in the root `README.md`.

The integration test `tests/integration/test_ml_engine_bundle.py` runs only when `MODEL_BUNDLE_PATH` points to a bundle folder, and classifies `ML_SAMPLE_IMAGE` when that file exists.

## Tests

```bash
docker compose run --rm --build backend-test   # from the repo root: full suite, own test database
```

Outside Docker: `pip install -e ".[dev]"`, then `pytest tests/` from `app/backend`. The integration tests run only when `TEST_DATABASE_URL` points to a database whose name contains `test` (they wipe it); `DATABASE_URL` is never used, so `ether_db` stays intact. Details in `docs/deployment.md` (Tests).
