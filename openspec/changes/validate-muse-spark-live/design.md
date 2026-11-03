# Design

## Context

See proposal.md, Why. Current state:

- `MetaModelLLM` (`adapters/outbound/llm/meta_model_llm.py`) posts to `/chat/completions` with `model`, `messages`, and either `response_format.json_schema` (`strict: true`) or `stream: true`. It sends no token bound and no reasoning effort. Images go on the last message as `data:image/png;base64,...`.
- The stored study image is the uploaded JPEG or PNG with its metadata removed, not re-encoded (`application/image_upload.py`). So the `image/png` label is wrong for JPEG uploads.
- The stream reader takes `choices[0].delta.content` from `data: ` lines. With `stream_options.include_usage`, Meta's schema says the final chunk has `choices: []` and a `usage` object. The current `chunk.get("choices", [{}])[0]` would raise `IndexError` on it.
- `CappedLLM.stream` (`spend_cap.py`) always adds a character estimate (4 characters a token, plus 1,500 for an image). `complete_json` usage reaches the cap through `on_usage=cap.record_usage`.
- `GenerateClinicalReport._draft` already treats a `ValidationError` as an invalid answer and retries once. So empty `content` caused by running out of tokens fails the report instead of crashing. It does waste the retry on the same token budget.
- What was checked so far (2026-11-03):
  - `GET /v1/models` with the key answers 200 and lists `muse-spark-1.3`.
  - Meta's pricing page gives USD 1.25 input, 0.15 cached input and 4.25 output per 1M tokens, and 3,000 requests and 4M tokens per minute per team.
  - The schema page documents `max_completion_tokens` (visible + reasoning; `max_tokens` is deprecated), `reasoning_effort` (`none` | `minimal` | `low` | `medium` | `high` | `xhigh` | `max`) and `usage.completion_tokens_details.reasoning_tokens`.

Findings of the live raw probe (2026-11-03, `muse-spark-1.3`, two runs plus one `minimal` stream, public `Te-gl_1.jpg` at 354x442):

- **Strict `json_schema` accepts `$defs`/`$ref`**: `GroundedClinicalReportOutput`'s schema was accepted as sent, and both reports (`low`, `medium`) validated against it. No schema transform is needed.
- **JPEG data labelled `image/png` is accepted** (HTTP 200, same prompt tokens). The wrong label is therefore not a live failure. It is still fixed, so the request states the truth and does not depend on Meta's tolerance.
- **An image costs about 184 prompt tokens** at this size (204 with the image vs about 20 for the same text alone), far below the spend cap's 1,500 estimate.
- **Reasoning dominates output**. With no effort set, a two-sentence text answer used 866 to 982 reasoning tokens out of 924 to 1,045 completion tokens. `completion_tokens` includes `reasoning_tokens`.
- **Streaming sends no reasoning deltas**, so the reader waits in silence. Time to first token for the real chat prompt with the image:

  | Effort | First token | Completion (reasoning) tokens |
  |---|---|---|
  | none set | 46.6 s | 4,494 (4,231) |
  | `low` | 7.8 s | 744 (679) |
  | `minimal` | 2.7 s | 242 (163) |

  The first sample (no effort set) reused the report's system prompt, so it is only an upper bound.
- **`none` is rejected** for `muse-spark-1.3`: HTTP 400 `invalid_request_error`, "Supported values: [minimal, low, medium, high, xhigh, max]".
- **Streamed usage arrives on the final content chunk**, the one with `finish_reason: "stop"`, followed by `data: [DONE]`. No separate `choices: []` chunk was sent, despite the schema page. Lines are `data: {...}` followed by a blank line.
- **Report at `low`**: 13.6 to 16.4 s, 1,174 to 1,659 completion tokens, about USD 0.007 to 0.009. At `medium`: 21.7 to 26.0 s, 2,165 to 2,703 tokens, about USD 0.011 to 0.013, with no gain the schema check can see.
- **Chosen defaults**:
  - report effort `low`;
  - chat effort `minimal` (decided with the user after the table above);
  - `max_completion_tokens` 4096, about 2.5 times the largest report completion seen at `low`, and well above every chat completion seen at `minimal`.

## Goals / Non-Goals

**Goals:**
- Measure each capability once against the real API before changing code, so every fix copies an observed response and not another reading of the docs.
- Keep the adapter's public surface (`LLMPort`) unchanged. The fixes are internal or configuration.

**Non-Goals:**
- Running live calls in CI or in `pytest`.
- Tuning prompts or report quality.

## Decisions

### 1. Probe first with raw HTTP, then confirm through the adapter

`scripts/probe_muse_spark.py` runs in two stages.

1. **Raw**: four `httpx` calls built by hand:
   - text
   - image + text
   - `json_schema` with `GroundedClinicalReportOutput`'s schema, so we learn whether strict mode accepts its `$defs`/`$ref`
   - stream with `include_usage`

   Each call records status, latency, raw `usage`, `finish_reason`, and the first bytes of the response.
2. **Adapter**: after the fixes, `MetaModelLLM.complete_json` with the image, and `MetaModelLLM.stream`. This proves the adapter's own requests work.

Alternative considered: only call the adapter. Rejected because a failure would show the adapter's error mapping ("LLM_REQUEST_FAILED 502") instead of Meta's error body, which is what tells us what to fix.

### 2. A script, not an integration test

The probe is a script run by hand. It reads the key from `app/backend/.env` and prints a Markdown table to paste into the README. Alternative considered: a `tests/integration` test skipped without a key. Rejected because the integration suite runs whenever a developer has a full `.env`, so it would spend credits on every run. The spec also requires that no test sends a request.

### 3. Media type detected from magic bytes

`\xFF\xD8\xFF` is JPEG and `\x89PNG\r\n\x1a\n` is PNG. Anything else raises a programming error, because uploads are already restricted to those two formats. Alternatives considered:
- Pass the MIME type through `LLMPort`: rejected because it changes the port and its callers for information the bytes already carry.
- Re-encode everything to PNG: rejected because it changes the pixels the classifier saw.

### 4. Token bound and one reasoning effort per use, as settings

Three settings:
- `META_MODEL_MAX_COMPLETION_TOKENS`: a positive int, default 4096.
- `META_MODEL_REPORT_REASONING_EFFORT`: default `low`.
- `META_MODEL_CHAT_REASONING_EFFORT`: default `minimal`.

Both efforts are a `Literal` of the values the model accepts (`minimal` through `max`; `none` is rejected live). The factory passes them to `MetaModelLLM`:
- `complete_json`, used only by the report, sends the bound and the report effort;
- `stream`, used only by the chat, sends the bound and the chat effort.

The effort follows the adapter method, so `LLMPort` does not change.

Why two efforts: the stream emits nothing while the model reasons. At the report's effort (`low`) the chat would wait about 8 s before its first word, against about 3 s at `minimal` (see Context). The report needs a valid, grounded JSON answer more than speed, and `low` delivered that.

Alternatives considered:
- One shared effort: rejected because the chat would be slow or the report untested.
- An effort argument on `LLMPort`: rejected because it changes the port and both callers when the method already identifies the use.
- Hard-coded constants: rejected because cost and latency are what the demo will most likely need to tune without a redeploy.

### 5. Stream usage reported through the existing listener, estimate as fallback

The stream reader:
- sends `stream_options: {"include_usage": true}`;
- takes `usage` from whichever chunk carries it: live, that is the final content chunk; per the docs it may be a chunk with `choices: []`;
- reads text only from chunks that have choices;
- calls `on_usage` exactly once per accepted stream, in a `finally`. It passes the measured usage when Meta sent it. Otherwise, for example when the reader stops early or the connection drops, it passes the character estimate that `CappedLLM` used to compute: 4 characters a token, plus an image estimate now set from the measured ~184 tokens.

`CappedLLM.stream` then only checks the limits, and the cap counts streams through the same listener as `complete_json`. Only the adapter knows whether usage arrived, so it is the one that decides measured or estimated.

Alternatives considered:
- A per-call flag that tells `CappedLLM` usage arrived: rejected because the listener is shared by concurrent calls, so the flag would need a context variable across async generators.
- Having `stream` yield a usage sentinel: rejected because it changes what `LLMPort.stream` yields to every caller.

### 6. Cost computed in the probe, not in the app

`cost = (prompt - cached) * 1.25e-6 + cached * 0.15e-6 + completion * 4.25e-6`, where `completion` includes reasoning tokens. The prices are constants in the probe, together with the date they were read. The app keeps counting tokens, not dollars. The spend cap's unit does not change.

## Risks / Trade-offs

- [Strict `json_schema` rejects `$defs`/`$ref`] → Checked live: it was accepted, so no transform is needed.
- [`minimal` chat answers are shallower] → One live answer was on-topic, cited evidence keys and stated uncertainty. It is a single sample; the effort is a setting, so it can be raised without a redeploy.
- [Reasoning tokens bill at a different rate than output, or not at all] → The cost column states the assumption that they bill as output. The README marks it unconfirmed until the Billing page shows the run.
- [Image token count varies with resolution] → Probe a dataset image at its stored size. Replace the 1,500 image estimate in the spend cap only if the measured number differs a lot, and record the measured number.
- [Meta may use the probe's content] → Only public dataset images and invented notes, per the spec.
- [The pasted key leaks] → It is stored only in the gitignored `.env`, the probe never prints it, and it is rotated after the run.
- [One run is a single sample of latency] → The README labels the numbers as one run on a date, not a benchmark.

## Migration Plan

Nothing to migrate. The new settings have defaults, so existing `.env` files keep working. Rollback is a revert of the adapter commit. The README results stay valid as a record.

## Open Questions

- Does `include_usage` also report usage when the client stops reading early? The fallback estimate covers it either way.
