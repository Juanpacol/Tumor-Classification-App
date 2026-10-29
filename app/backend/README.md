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

