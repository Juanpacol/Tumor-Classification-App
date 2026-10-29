import json

import httpx
import pytest
from pydantic import BaseModel

from adapters.outbound.llm.meta_model_llm import MetaModelLLM
from domain.errors import DomainError


class Address(BaseModel):
    street: str
    city: str


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.meta.ai/v1"
    )


def _llm(handler, api_key: str | None = "test-key") -> MetaModelLLM:
    return MetaModelLLM(
        api_key=api_key,
        model="muse-spark-1.3",
        client=_client(handler),
        retry_backoff_seconds=0,
    )


def _completion_response(content: str, usage: dict | None = None) -> httpx.Response:
    body = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


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
async def test_complete_json_success_with_image():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        content = payload["messages"][-1]["content"]
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        return _completion_response(
            json.dumps({"street": "1 Hacker Wy", "city": "Menlo Park"})
        )

    llm = _llm(handler)
    result = await llm.complete_json(
        [{"role": "user", "content": "hi"}], Address, image=b"fake-png-bytes"
    )

    assert result == Address(street="1 Hacker Wy", city="Menlo Park")


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
