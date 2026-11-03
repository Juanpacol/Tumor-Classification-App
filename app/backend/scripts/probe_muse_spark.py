"""Live check of the Meta Model API (Muse Spark), run by hand with a real key.

Stage ``raw`` builds every request by hand so Meta's own answer, or error body,
is visible. Stage ``adapter`` repeats the report and chat calls through
``build_llm(Settings())`` to prove the adapter's requests work.

Only public dataset images and invented notes are sent: while free credits are
in use Meta may use the content. The key is read from ``.env`` and never printed.

Usage, from ``app/backend``::

    python scripts/probe_muse_spark.py raw --save-dir tests/fixtures/meta_model
    python scripts/probe_muse_spark.py adapter
"""

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import httpx

from adapters.outbound.llm.factory import build_llm
from adapters.outbound.llm.meta_model_llm import BASE_URL
from application.chat_prompt import build_messages as build_chat_messages
from application.evidence_bundle import BuildEvidenceBundle, EvidenceBundle
from application.grounded_report import GroundedClinicalReportOutput
from application.report_prompt import build_messages as build_report_messages
from domain.models import Prediction, Study, StudyStatus
from settings import Settings

PRICES_READ_ON = "2026-11-03"
USD_PER_INPUT_TOKEN = 1.25 / 1_000_000
USD_PER_CACHED_INPUT_TOKEN = 0.15 / 1_000_000
USD_PER_OUTPUT_TOKEN = 4.25 / 1_000_000
PROBE_MAX_COMPLETION_TOKENS = 16_000

DEFAULT_IMAGE = (
    BACKEND_DIR.parent
    / "ml_engine"
    / "src"
    / "data"
    / "Test"
    / "glioma"
    / "Te-gl_1.jpg"
)
INVENTED_NOTES = "Invented case for an API test: adult with three weeks of headache."
CHAT_QUESTION = "In two sentences, what should I check first on this image?"
STREAM_EFFORTS = ("none", "low")


@dataclass
class CallResult:
    """What one live call returned."""

    capability: str
    status: int
    latency_s: float
    usage: dict[str, Any] | None
    finish_reason: str | None = None
    answer: str = ""
    first_token_s: float | None = None
    raw: list[str] = field(default_factory=list)

    @property
    def cost_usd(self) -> float | None:
        if not self.usage:
            return None
        prompt = self.usage.get("prompt_tokens", 0)
        cached = _cached_tokens(self.usage)
        completion = self.usage.get("completion_tokens", 0)
        return (
            (prompt - cached) * USD_PER_INPUT_TOKEN
            + cached * USD_PER_CACHED_INPUT_TOKEN
            + completion * USD_PER_OUTPUT_TOKEN
        )


def _cached_tokens(usage: dict[str, Any]) -> int:
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0


def _reasoning_tokens(usage: dict[str, Any]) -> int:
    return (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0


def invented_case() -> tuple[Study, Prediction, EvidenceBundle]:
    """An invented study and prediction: nothing here comes from a patient."""
    study = Study(
        id="probe-study",
        file_key="studies/probe-study",
        original_format="jpeg",
        status=StudyStatus.COMPLETED,
        created_at=datetime.now(UTC),
        clinical_notes=INVENTED_NOTES,
    )
    prediction = Prediction(
        study_id=study.id,
        label="glioma",
        probabilities={
            "glioma": 0.81,
            "meningioma": 0.11,
            "pituitary": 0.05,
            "notumor": 0.03,
        },
        confidence=0.81,
        needs_expert_triage=False,
        quality={"snr": 14.2, "passed": True},
        top_features=[{"name": "glcm_contrast", "value": 0.42}],
        model_version="probe",
    )
    return study, prediction, BuildEvidenceBundle().execute(prediction)


def invented_report_messages() -> list[dict[str, Any]]:
    """The app's real report prompt for the invented case."""
    return build_report_messages(*invented_case())


def invented_chat_messages() -> list[dict[str, Any]]:
    """The app's real chat prompt for the invented case, with no report or history."""
    study, prediction, bundle = invented_case()
    return build_chat_messages(study, prediction, bundle, None, [], CHAT_QUESTION)


def strict_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": GroundedClinicalReportOutput.__name__,
            "strict": True,
            "schema": GroundedClinicalReportOutput.model_json_schema(),
        },
    }


def with_image(
    messages: list[dict[str, Any]], image: bytes, media_type: str
) -> list[dict[str, Any]]:
    encoded = base64.b64encode(image).decode("ascii")
    last = dict(messages[-1])
    last["content"] = [
        {"type": "text", "text": last["content"]},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{encoded}"},
        },
    ]
    return [*messages[:-1], last]


async def post(
    client: httpx.AsyncClient, capability: str, payload: dict[str, Any]
) -> CallResult:
    start = time.perf_counter()
    response = await client.post("/chat/completions", json=payload)
    latency = time.perf_counter() - start
    result = CallResult(capability, response.status_code, latency, None)
    result.raw = [response.text]
    if response.status_code != 200:
        result.answer = response.text
        return result
    body = response.json()
    choice = body["choices"][0]
    result.usage = body.get("usage")
    result.finish_reason = choice.get("finish_reason")
    result.answer = choice["message"].get("content") or ""
    return result


async def post_stream(
    client: httpx.AsyncClient, capability: str, payload: dict[str, Any]
) -> CallResult:
    start = time.perf_counter()
    result = CallResult(capability, 0, 0.0, None)
    async with client.stream(
        "POST",
        "/chat/completions",
        json=payload,
        headers={"Accept": "text/event-stream"},
    ) as response:
        result.status = response.status_code
        async for line in response.aiter_lines():
            result.raw.append(line)
            data = line.removeprefix("data:").strip()
            if not line.startswith("data:") or data == "[DONE]":
                continue
            chunk = json.loads(data)
            if chunk.get("usage"):
                result.usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    if result.first_token_s is None:
                        result.first_token_s = time.perf_counter() - start
                    result.answer += delta
                result.finish_reason = choice.get("finish_reason") or (
                    result.finish_reason
                )
    result.latency_s = time.perf_counter() - start
    return result


async def run_raw(settings: Settings, image: bytes, save_dir: Path | None) -> None:
    model = settings.meta_model_id
    base = {"model": model, "max_completion_tokens": PROBE_MAX_COMPLETION_TOKENS}
    report = invented_report_messages()
    describe = [
        {
            "role": "user",
            "content": "This is a public MRI test image. Describe it in two sentences.",
        }
    ]
    headers = {"Authorization": f"Bearer {settings.meta_model_api_key}"}
    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(
        base_url=BASE_URL, headers=headers, timeout=timeout
    ) as client:
        results = [
            await post(
                client,
                "text",
                {
                    **base,
                    "messages": [
                        {
                            "role": "user",
                            "content": "In two sentences, what is a T1-weighted MRI?",
                        }
                    ],
                },
            ),
            await post(
                client,
                "image (jpeg)",
                {**base, "messages": with_image(describe, image, "image/jpeg")},
            ),
            await post(
                client,
                "image (jpeg labelled png)",
                {**base, "messages": with_image(describe, image, "image/png")},
            ),
        ]
        for effort in ("low", "medium"):
            results.append(
                await post(
                    client,
                    f"structured JSON ({effort})",
                    {
                        **base,
                        "reasoning_effort": effort,
                        "response_format": strict_schema(),
                        "messages": with_image(report, image, "image/jpeg"),
                    },
                )
            )
        chat = with_image(invented_chat_messages(), image, "image/jpeg")
        for effort in STREAM_EFFORTS:
            results.append(
                await post_stream(
                    client,
                    f"streaming ({effort})",
                    {
                        **base,
                        "reasoning_effort": effort,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "messages": chat,
                    },
                )
            )
    for result in results:
        print_detail(result)
        if result.capability.startswith("structured JSON"):
            print_report_check(result)
    print_table(model, results)
    if save_dir:
        save_fixtures(save_dir, results)


class UsageCollector(logging.Handler):
    """Collects the usage the adapter logs, so the probe never reaches into it."""

    def __init__(self) -> None:
        super().__init__()
        self.usages: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        usage = getattr(record, "usage", None)
        if usage:
            self.usages.append(usage)


async def run_adapter(settings: Settings, image: bytes) -> None:
    collector = UsageCollector()
    adapter_logger = logging.getLogger("adapters.outbound.llm.meta_model")
    adapter_logger.addHandler(collector)
    adapter_logger.setLevel(logging.INFO)
    llm = build_llm(settings)
    results: list[CallResult] = []
    try:
        start = time.perf_counter()
        output = await llm.complete_json(
            invented_report_messages(), GroundedClinicalReportOutput, image=image
        )
        results.append(
            CallResult(
                "adapter structured JSON",
                200,
                time.perf_counter() - start,
                collector.usages[-1] if collector.usages else None,
                answer=output.model_dump_json()[:300],
            )
        )
        chat = invented_chat_messages()
        before = len(collector.usages)
        start = time.perf_counter()
        result = CallResult("adapter streaming", 200, 0.0, None)
        async for chunk in llm.stream(chat, image=image):
            if result.first_token_s is None:
                result.first_token_s = time.perf_counter() - start
            result.answer += chunk
        result.latency_s = time.perf_counter() - start
        if len(collector.usages) > before:
            result.usage = collector.usages[-1]
        results.append(result)
    finally:
        await llm.aclose()
    for result in results:
        print_detail(result)
    print_table(settings.meta_model_id, results)


def print_detail(result: CallResult) -> None:
    print(f"\n=== {result.capability}: HTTP {result.status}, {result.latency_s:.2f}s")
    if result.first_token_s is not None:
        print(f"first token after {result.first_token_s:.2f}s")
    print(f"finish_reason: {result.finish_reason}")
    print(f"usage: {json.dumps(result.usage)}")
    print(f"answer: {result.answer[:300]!r}")


def print_report_check(result: CallResult) -> None:
    if result.status != 200:
        return
    try:
        GroundedClinicalReportOutput.model_validate_json(result.answer)
        print("report: matches GroundedClinicalReportOutput")
    except ValueError as error:
        print(f"report: does NOT match the schema ({type(error).__name__})")


def print_table(model: str, results: list[CallResult]) -> None:
    print(f"\nModel `{model}`, prices read on {PRICES_READ_ON}.\n")
    print(
        "| Capability | HTTP | Latency (s) | First token (s) | Prompt | Cached "
        "| Completion | Reasoning | USD |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        usage = r.usage or {}
        cost = r.cost_usd
        first = f"{r.first_token_s:.2f}" if r.first_token_s is not None else "-"
        print(
            f"| {r.capability} | {r.status} | {r.latency_s:.2f} | {first} "
            f"| {usage.get('prompt_tokens', '-')} "
            f"| {_cached_tokens(usage) if usage else '-'} "
            f"| {usage.get('completion_tokens', '-')} "
            f"| {_reasoning_tokens(usage) if usage else '-'} "
            f"| {f'{cost:.5f}' if cost is not None else '-'} |"
        )


def save_fixtures(save_dir: Path, results: list[CallResult]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    by_capability = {r.capability: r for r in results}
    report = by_capability["structured JSON (low)"]
    stream = by_capability[f"streaming ({STREAM_EFFORTS[-1]})"]
    if report.status == 200:
        body = json.loads(report.raw[0])
        (save_dir / "chat_completion_report.json").write_text(
            json.dumps(body, indent=2) + "\n", encoding="utf-8"
        )
    if stream.status == 200:
        (save_dir / "chat_completion_stream.sse").write_text(
            "\n".join(stream.raw) + "\n", encoding="utf-8"
        )
    print(f"\nFixtures written to {save_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["raw", "adapter"])
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--save-dir", type=Path, help="raw stage: write fixtures here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(_env_file=BACKEND_DIR / ".env")
    if not settings.meta_model_api_key:
        sys.exit("META_MODEL_API_KEY is not set in app/backend/.env")
    if settings.meta_model_id.endswith("-contributor"):
        sys.exit("Refusing a -contributor model: Meta trains on its content")
    image = args.image.read_bytes()
    if args.stage == "raw":
        asyncio.run(run_raw(settings, image, args.save_dir))
    else:
        asyncio.run(run_adapter(settings, image))


if __name__ == "__main__":
    main()
