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
