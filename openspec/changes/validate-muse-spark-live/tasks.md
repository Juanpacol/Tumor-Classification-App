# Tasks

## 1. Key and safe inputs

- [x] 1.1 Put the key in `app/backend/.env` as `META_MODEL_API_KEY`, copied from `.env.example`. Verify `git check-ignore app/backend/.env` matches and `git status` does not list the file
- [x] 1.2 Pick the probe inputs and verify both:
  - one public JPEG from `app/ml_engine/src/data/Test/` (for example `glioma/Te-gl_1.jpg`), and that it is a JPEG at its stored size;
  - an invented note with no names, dates or identifiers, and an invented classifier output.

## 2. Raw probe (design D1, D2, D6)

- [x] 2.1 Write `app/backend/scripts/probe_muse_spark.py`:
  - It reads `META_MODEL_API_KEY` and `META_MODEL_ID` from `Settings` and refuses any `-contributor` model id.
  - It builds each request by hand with `httpx`, never prints the key, and holds the prices (1.25 / 0.15 / 4.25 USD per 1M, read 2026-11-03) as constants.
  - For each call it prints status, latency, raw `usage`, `finish_reason` and the first 300 characters of the answer, and ends with a Markdown table (capability, latency, prompt, cached, completion, reasoning, USD).

  Verify `python scripts/probe_muse_spark.py --help` runs without a network call.
- [x] 2.2 Run stage 1: text, image + text (labelled JPEG and mislabelled PNG), `json_schema` (strict) with `GroundedClinicalReportOutput`'s schema at `reasoning_effort` `low` and `medium`, and the real chat prompt streamed with `stream_options.include_usage` at `none`, `minimal` and `low`. Verify each call returns 200, or Meta's error body is captured.
- [x] 2.3 Save the captured raw responses, with any content that is not ours removed, as fixtures in `app/backend/tests/fixtures/meta_model/` (a completion body with `usage`, and an SSE transcript ending with the usage-carrying final chunk and `[DONE]`; the live stream sent no separate `choices: []` chunk). Verify the files parse as JSON or SSE.
- [x] 2.4 Record the findings in the design's Context:
  - whether strict mode accepted `$defs`/`$ref`;
  - whether JPEG data labelled `image/png` was accepted;
  - the image's prompt tokens;
  - the chosen defaults for effort and `max_completion_tokens`.

  Verify design.md states each one.

## 3. Request fixes in the adapter (design D3, D4)

- [x] 3.1 Detect the image media type from the magic bytes in `meta_model_llm.py` (JPEG → `image/jpeg`, PNG → `image/png`, anything else raises). Verify new `test_meta_model_llm.py` cases: a JPEG fixture is sent as `data:image/jpeg`, a PNG as `data:image/png`, and unknown bytes raise.
- [x] 3.2 Add `meta_model_max_completion_tokens` (positive int, default 4096), `meta_model_report_reasoning_effort` (default `low`) and `meta_model_chat_reasoning_effort` (default `minimal`) to `settings.py`. Both efforts are a `Literal` of `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Pass them through `factory.py` into `MetaModelLLM`: `complete_json` sends the bound and the report effort, and `stream` sends the bound and the chat effort. Verify:
  - `test_settings.py` covers the defaults, plus rejection of 0 and of `none`;
  - `test_meta_model_llm.py` asserts each payload carries the bound and its own effort;
  - `test_llm_wiring.py` asserts the factory passes all three.
- [x] 3.3 If 2.4 found that strict mode rejects `$defs`/`$ref`, inline the definitions before sending and verify a test asserts the sent schema has no `$ref`. Otherwise mark this task done with "not needed: accepted live". Not needed: accepted live (2026-11-03).
- [x] 3.4 Replace the hand-written completion bodies in `test_meta_model_llm.py` with the fixture from 2.3, and verify the suite passes: `cd app/backend && python -m pytest tests/test_meta_model_llm.py -v`.
- [x] 3.5 Add the three settings, with a comment each, to `app/backend/.env.example`. Verify the file lists them.

## 4. Stream usage and the spend cap (design D5)

- [x] 4.1 In `MetaModelLLM.stream`:
  - send `stream_options: {"include_usage": true}`;
  - take `usage` from whichever chunk carries it, and read text only from chunks with choices;
  - call `on_usage` exactly once per accepted stream: the measured usage, or the character estimate when none arrived (early stop, dropped connection).

  Verify tests that:
  - replay the SSE fixture from 2.3 and get every text chunk plus one listener call with the fixture's usage;
  - replay a stream whose usage is on a `choices: []` chunk;
  - abandon a stream early and get one estimated usage.
- [x] 4.2 In `spend_cap.py`, reduce `CappedLLM.stream` to the limit check plus pass-through. The stream's tokens now reach the cap through the adapter's listener, so they are counted once. Move the estimate tests from `test_spend_cap.py` to `test_meta_model_llm.py`, and verify `test_spend_cap.py` asserts:
  - a capped stream adds no tokens on its own;
  - a stream over the limits still fails when iteration starts;
  - `complete_json` counting is unchanged.
- [x] 4.3 If 2.2 measured image tokens far from 1,500, update `IMAGE_TOKEN_ESTIMATE` to the measured value and verify `test_spend_cap.py` still passes. Otherwise leave it and say so in the README.
- [x] 4.4 Update the README "Spend cap" section: streams count measured usage, and the estimate is only a fallback, with the measured image token figure. Verify the "guesses until the live validation (AB-161)" sentence is gone.

## 5. Confirm through the adapter and document

- [x] 5.1 Run stage 2 of the probe:
  - `MetaModelLLM.complete_json` with the JPEG and `GroundedClinicalReportOutput`;
  - `MetaModelLLM.stream` with the same image, both built through `build_llm(Settings())`.

  Verify the report validates, the stream yields text, and the probe table shows usage for both.
- [x] 5.2 Rewrite the README "Muse Spark / Meta Model API" section:
  - correct the image claim (real JPEG/PNG type, stored size, not "only PNG ≤1024px");
  - document `max_completion_tokens`, `reasoning_effort` and `include_usage`;
  - add a "Live validation (2026-11-xx)" table with one row per capability (text, image, structured JSON, streaming) and one per effort level, with latency, prompt, cached, completion, reasoning tokens and USD;
  - state the prices and the date they were read;
  - list the adapter fixes;
  - mark "reasoning tokens assumed billed as output" as unconfirmed;
  - say the numbers are one run, not a benchmark.

  Also replace the "Missing (AB-161)" sentence in the Chat section with what is still missing (manual review of 10 reports). Verify every row has every column filled.
- [x] 5.3 Verify the full backend suite passes with no network access to Meta (`cd app/backend && python -m pytest tests/ -v`, with `META_MODEL_API_KEY` unset), and verify a search of `app/` and `openspec/` for the key value read from `app/backend/.env` (never typed into a command or file) finds it only in `.env`.

## Workflow follow-up

- Rotate the Meta API key in the console once the validation run is done (it was pasted in a chat), and update `app/backend/.env` and the Cloud Run secret.
- Commit as `AB-161: ...` and open the PR to `develop` with the `create-commit` and `create-pull-request` skills.
- Archive the change after review.
