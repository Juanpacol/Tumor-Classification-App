import copy
import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from adapters.outbound.llm.meta_model_llm import IMAGE_TOKEN_ESTIMATE, MetaModelLLM
from application.grounded_report import GroundedClinicalReportOutput
from domain.errors import DomainError
from tests.factories import image_bytes

FIXTURES = Path(__file__).parent / "fixtures" / "meta_model"
LIVE_COMPLETION = json.loads(
    (FIXTURES / "chat_completion_report.json").read_text(encoding="utf-8")
)
LIVE_STREAM = (FIXTURES / "chat_completion_stream.sse").read_text(encoding="utf-8")


class Address(BaseModel):
    street: str
    city: str


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.meta.ai/v1"
    )


def _llm(handler, api_key: str | None = "test-key", on_usage=None) -> MetaModelLLM:
    return MetaModelLLM(
        api_key=api_key,
        model="muse-spark-1.3",
        max_completion_tokens=4096,
        report_reasoning_effort="low",
        chat_reasoning_effort="minimal",
        client=_client(handler),
        retry_backoff_seconds=0,
        on_usage=on_usage,
    )


def _completion_response(content: str, usage: dict | None = None) -> httpx.Response:
    """The envelope Meta returned live, with this test's content and usage."""
    body = copy.deepcopy(LIVE_COMPLETION)
    body["choices"][0]["message"]["content"] = content
    if usage is None:
        del body["usage"]
    else:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def _sse_response(body: str) -> httpx.Response:
    return httpx.Response(
        200, content=body, headers={"Content-Type": "text/event-stream"}
    )


@pytest.mark.asyncio
async def test_complete_json_success_without_image():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        return _completion_response(
            json.dumps({"street": "1 Hacker Wy", "city": "Menlo Park"})
        )

    llm = _llm(handler)
    result = await llm.complete_json([{"role": "user", "content": "hi"}], Address)

    assert result == Address(street="1 Hacker Wy", city="Menlo Park")


@pytest.mark.asyncio
@pytest.mark.parametrize(("fmt", "media_type"), [("PNG", "png"), ("JPEG", "jpeg")])
async def test_an_image_is_sent_with_its_real_media_type(fmt: str, media_type: str):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        content = payload["messages"][-1]["content"]
        assert content[0] == {"type": "text", "text": "hi"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith(
            f"data:image/{media_type};base64,"
        )
        return _completion_response(
            json.dumps({"street": "1 Hacker Wy", "city": "Menlo Park"})
        )

    llm = _llm(handler)
    result = await llm.complete_json(
        [{"role": "user", "content": "hi"}], Address, image=image_bytes(fmt)
    )

    assert result == Address(street="1 Hacker Wy", city="Menlo Park")


@pytest.mark.asyncio
async def test_an_image_that_is_neither_jpeg_nor_png_is_refused_before_sending():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not call the network")

    with pytest.raises(ValueError, match="JPEG and PNG"):
        await _llm(handler).complete_json(
            [{"role": "user", "content": "hi"}], Address, image=b"GIF89a..."
        )


@pytest.mark.asyncio
async def test_a_report_request_bounds_its_tokens_and_uses_the_report_effort():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["max_completion_tokens"] == 4096
        assert payload["reasoning_effort"] == "low"
        assert payload["response_format"]["json_schema"]["strict"] is True
        return _completion_response('{"street": "Main", "city": "Bogota"}')

    await _llm(handler).complete_json([{"role": "user", "content": "x"}], Address)


@pytest.mark.asyncio
async def test_a_chat_stream_bounds_its_tokens_uses_the_chat_effort_and_asks_for_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["max_completion_tokens"] == 4096
        assert payload["reasoning_effort"] == "minimal"
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        return _sse_response("data: [DONE]\n")

    [c async for c in _llm(handler).stream([{"role": "user", "content": "x"}])]


@pytest.mark.asyncio
async def test_complete_json_retries_once_on_429_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return _completion_response(
            json.dumps({"street": "1 Hacker Wy", "city": "Menlo Park"})
        )

    llm = _llm(handler)
    result = await llm.complete_json([{"role": "user", "content": "hi"}], Address)

    assert calls["count"] == 2
    assert result == Address(street="1 Hacker Wy", city="Menlo Park")


@pytest.mark.asyncio
async def test_complete_json_does_not_retry_on_auth_failure():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": "invalid key"})

    llm = _llm(handler)
    with pytest.raises(DomainError) as exc_info:
        await llm.complete_json([{"role": "user", "content": "hi"}], Address)

    assert calls["count"] == 1
    assert exc_info.value.code == "LLM_AUTH_FAILED"
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_complete_json_raises_on_timeout_after_one_retry():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    llm = _llm(handler)
    with pytest.raises(DomainError) as exc_info:
        await llm.complete_json([{"role": "user", "content": "hi"}], Address)

    assert exc_info.value.code == "LLM_TIMEOUT"
    assert exc_info.value.status_code == 504


@pytest.mark.asyncio
async def test_missing_api_key_raises_llm_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not call the network without an API key")

    llm = _llm(handler, api_key=None)
    with pytest.raises(DomainError) as exc_info:
        await llm.complete_json([{"role": "user", "content": "hi"}], Address)

    assert exc_info.value.code == "LLM_UNAVAILABLE"
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_stream_yields_text_chunks():
    sse_body = (
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n'
        'data: {"choices": [{"delta": {"content": " world"}}]}\n'
        "data: [DONE]\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse_body, headers={"Content-Type": "text/event-stream"}
        )

    llm = _llm(handler)
    chunks = [chunk async for chunk in llm.stream([{"role": "user", "content": "hi"}])]

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_stream_skips_malformed_chunks():
    sse_body = (
        "data: not-json\n"
        'data: {"choices": [{"delta": {"content": "ok"}}]}\n'
        "data: [DONE]\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse_body, headers={"Content-Type": "text/event-stream"}
        )

    llm = _llm(handler)
    chunks = [chunk async for chunk in llm.stream([{"role": "user", "content": "hi"}])]

    assert chunks == ["ok"]


async def test_usage_is_handed_to_the_listener():
    seen: list[dict | None] = []
    usage = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}

    def handler(request: httpx.Request) -> httpx.Response:
        return _completion_response('{"street": "Main", "city": "Bogota"}', usage)

    llm = _llm(handler, on_usage=seen.append)

    await llm.complete_json([{"role": "user", "content": "x"}], Address)

    assert seen == [usage]


async def test_a_response_without_usage_still_notifies_the_listener():
    seen: list[dict | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return _completion_response('{"street": "Main", "city": "Bogota"}')

    llm = _llm(handler, on_usage=seen.append)

    await llm.complete_json([{"role": "user", "content": "x"}], Address)

    assert seen == [None]


@pytest.mark.asyncio
async def test_the_live_report_response_parses_into_the_grounded_report_schema():
    seen: list[dict | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"]["json_schema"]["name"] == (
            "GroundedClinicalReportOutput"
        )
        return httpx.Response(200, json=LIVE_COMPLETION)

    llm = _llm(handler, on_usage=seen.append)
    result = await llm.complete_json(
        [{"role": "user", "content": "x"}], GroundedClinicalReportOutput
    )

    assert isinstance(result, GroundedClinicalReportOutput)
    assert result.findings
    assert seen == [LIVE_COMPLETION["usage"]]


def _live_stream_text() -> str:
    return "".join(
        choice["delta"].get("content") or ""
        for line in LIVE_STREAM.splitlines()
        if line.startswith("data: {")
        for choice in json.loads(line.removeprefix("data: "))["choices"]
    )


@pytest.mark.asyncio
async def test_the_live_stream_yields_its_text_and_reports_its_usage_once():
    seen: list[dict | None] = []
    llm = _llm(lambda request: _sse_response(LIVE_STREAM), on_usage=seen.append)

    chunks = [c async for c in llm.stream([{"role": "user", "content": "x"}])]

    assert "".join(chunks) == _live_stream_text() != ""
    assert len(seen) == 1
    assert seen[0]["completion_tokens_details"]["reasoning_tokens"] > 0


@pytest.mark.asyncio
async def test_usage_on_a_chunk_without_choices_is_reported_and_yields_nothing():
    seen: list[dict | None] = []
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    text_chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
    sse_body = (
        f"data: {json.dumps(text_chunk)}\n\n"
        f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
        "data: [DONE]\n\n"
    )
    llm = _llm(lambda request: _sse_response(sse_body), on_usage=seen.append)

    chunks = [c async for c in llm.stream([{"role": "user", "content": "x"}])]

    assert chunks == ["ok"]
    assert seen == [usage]


@pytest.mark.asyncio
async def test_a_stream_without_usage_reports_an_estimate():
    seen: list[dict | None] = []
    text_chunk = {"choices": [{"delta": {"content": "a" * 40}}]}
    sse_body = f"data: {json.dumps(text_chunk)}\ndata: [DONE]\n"
    llm = _llm(lambda request: _sse_response(sse_body), on_usage=seen.append)

    [c async for c in llm.stream([{"role": "user", "content": "c" * 120}])]

    assert seen == [{"total_tokens": (120 + 40) // 4, "estimated": True}]


@pytest.mark.asyncio
async def test_an_image_adds_its_estimate_to_a_stream_without_usage():
    seen: list[dict | None] = []
    llm = _llm(lambda request: _sse_response("data: [DONE]\n"), on_usage=seen.append)

    messages = [{"role": "user", "content": ""}]
    [c async for c in llm.stream(messages, image=image_bytes("JPEG"))]

    assert seen == [{"total_tokens": IMAGE_TOKEN_ESTIMATE, "estimated": True}]


@pytest.mark.asyncio
async def test_a_stream_the_reader_abandons_reports_an_estimate_of_what_was_sent():
    seen: list[dict | None] = []
    llm = _llm(lambda request: _sse_response(LIVE_STREAM), on_usage=seen.append)

    stream = llm.stream([])
    first = await anext(stream)
    await stream.aclose()

    assert seen == [{"total_tokens": len(first) // 4, "estimated": True}]
