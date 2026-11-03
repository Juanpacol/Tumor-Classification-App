# Proposal

## Why

`MetaModelLLM` was written against Meta's docs and has only been tested with mocked HTTP (AB-161). Its request format, image input, structured output and streaming have never been checked against the real API. Reading the code against the published schema already shows gaps:

- Every image is labelled `image/png`, but the stored image is the uploaded JPEG or PNG, which is not re-encoded.
- No output-token limit is sent, so reasoning can use the whole budget and leave `content` empty.
- Streamed calls never ask for usage, so the spend cap guesses their tokens.

Nothing measures what a call costs either.

## What Changes

- Make one live call per capability (plain text, image, structured JSON, streaming) against `https://api.meta.ai/v1/chat/completions` with `muse-spark-1.3`. Use only images from the public dataset (`app/ml_engine/src/data/Test/`) and invented notes, because Meta may use content sent while free credits are in use. Record latency, input, cached, output and reasoning tokens, and cost for each call.
- Fix the adapter where the real API differs from the mocked assumptions. Already known:
  - Send the image's real MIME type (JPEG or PNG, detected from the bytes) instead of always `image/png`.
  - Send `max_completion_tokens` and `reasoning_effort`, both configurable. Their defaults are chosen from the measured calls.
  - Ask for usage on streams (`stream_options.include_usage`). Ignore the final usage chunk, which has an empty `choices` list. Hand the streamed usage to the spend cap.
  - The spend cap counts real usage for a stream when Meta reports it, and keeps the character estimate only when it does not, for example on an interrupted stream.
- Replace the mocked tests' invented payloads with the shapes the live calls actually returned.
- Rewrite the Muse Spark section of the backend README:
  - Remove claims that turned out false, such as "only ever sends PNG ≤1024px".
  - Add a dated results table (capability, latency, tokens, cost) and the prices it uses.
  - List the adapter fixes.
  - Mark what is still unconfirmed, such as whether reasoning tokens bill as output.
- Out of scope:
  - The `/v1/responses` endpoint, unless chat completions cannot do one of the four capabilities.
  - A plain-text method on `LLMPort`: the text call is made directly, because no use case needs it.
  - Manual review of 10 reports.
  - A spend counter kept in the database.

## Capabilities

### New Capabilities
- `llm-gateway`: how the backend talks to the hosted LLM. It covers the request it sends (output-token bound, reasoning effort, image media type), how streamed and non-streamed answers are read, how token usage is reported to the spend cap, and the rule that only public-dataset images and invented notes reach Meta while free credits are in use.

### Modified Capabilities
None. No existing spec covers the LLM.

## Impact

- **Backend** (`app/backend`):
  - `adapters/outbound/llm/meta_model_llm.py`: MIME type, token limit, reasoning effort, stream usage, usage chunk.
  - `adapters/outbound/llm/spend_cap.py`: measured stream usage, with the estimate as fallback.
  - `adapters/outbound/llm/factory.py` and `settings.py`: the two new settings.
  - `.env.example`.
  - Tests: `test_meta_model_llm.py`, `test_spend_cap.py`, `test_llm_wiring.py`, `test_settings.py`.
- **New script**: `app/backend/scripts/probe_muse_spark.py`. Run by hand with a real key. Not part of the test suite, so CI never spends credits.
- **Docs**: `app/backend/README.md`, Muse Spark section.
- **Cost**: about 10 calls at USD 1.25 / 4.25 per 1M input/output tokens, well under one dollar of the free credits.
- **Secrets**: the key lives only in the gitignored `app/backend/.env`. It was pasted in a chat, so it is rotated once the validation is done.
