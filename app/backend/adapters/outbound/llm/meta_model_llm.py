import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from domain.errors import DomainError

BASE_URL = "https://api.meta.ai/v1"
CHAT_COMPLETIONS_PATH = "/chat/completions"
CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 90.0
RETRY_BACKOFF_SECONDS = 0.5
RETRYABLE_STATUS_CODES = {429}
CHARACTERS_PER_TOKEN = 4
IMAGE_TOKEN_ESTIMATE = 200
MEDIA_TYPES = {b"\xff\xd8\xff": "image/jpeg", b"\x89PNG\r\n\x1a\n": "image/png"}

logger = logging.getLogger("adapters.outbound.llm.meta_model")


class MetaModelLLM:
    """LLMPort adapter for Muse Spark via the Meta Model API.

    Checked live on 2026-11-03 (see the backend README): Bearer auth,
    structured output via strict `response_format.json_schema`, and SSE
    streaming whose final content chunk carries `usage` when
    `stream_options.include_usage` is set. `complete_json` serves the report and
    `stream` the chat, so each sends its own reasoning effort.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        max_completion_tokens: int,
        report_reasoning_effort: str,
        chat_reasoning_effort: str,
        client: httpx.AsyncClient | None = None,
        retry_backoff_seconds: float = RETRY_BACKOFF_SECONDS,
        on_usage: Callable[[dict[str, Any] | None], None] | None = None,
    ) -> None:
        self._on_usage = on_usage
        self._api_key = api_key
        self._model = model
        self._max_completion_tokens = max_completion_tokens
        self._report_reasoning_effort = report_reasoning_effort
        self._chat_reasoning_effort = chat_reasoning_effort
        self._retry_backoff_seconds = retry_backoff_seconds
        self._client = client or httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(
                CONNECT_TIMEOUT_SECONDS,
                connect=CONNECT_TIMEOUT_SECONDS,
                read=READ_TIMEOUT_SECONDS,
            ),
        )

    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: type,
        image: bytes | None = None,
    ) -> Any:
        api_key = self._require_api_key()
        payload = {
            "model": self._model,
            "messages": self._build_messages(messages, image),
            "max_completion_tokens": self._max_completion_tokens,
            "reasoning_effort": self._report_reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            },
        }
        response = await self._send(self._headers(api_key), payload, stream=False)
        self._raise_for_error(response)
        body = response.json()
        self._log_usage(body.get("usage"))
        content = body["choices"][0]["message"]["content"]
        return schema.model_validate_json(content)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        image: bytes | None = None,
    ) -> AsyncIterator[str]:
        """Yield the answer's text chunks and report the call's usage once.

        The usage is Meta's when the stream carried it, and an estimate from
        the characters sent and received when it did not (an early stop or a
        dropped connection), so the spend cap always counts the call.
        """
        api_key = self._require_api_key()
        payload = {
            "model": self._model,
            "messages": self._build_messages(messages, image),
            "max_completion_tokens": self._max_completion_tokens,
            "reasoning_effort": self._chat_reasoning_effort,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        response = await self._send(
            self._headers(api_key, streaming=True), payload, stream=True
        )
        self._raise_for_error(response)
        usage: dict[str, Any] | None = None
        streamed_characters = 0
        try:
            async for chunk in self._chunks(response):
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        streamed_characters += len(delta)
                        yield delta
        finally:
            await response.aclose()
            self._log_usage(
                usage or _estimated_usage(messages, image, streamed_characters)
            )

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    async def _chunks(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue

    def _require_api_key(self) -> str:
        if not self._api_key:
            raise DomainError(
                code="LLM_UNAVAILABLE",
                message="Muse Spark API key is not configured",
                status_code=503,
            )
        return self._api_key

    @staticmethod
    def _headers(api_key: str, *, streaming: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if streaming:
            headers["Accept"] = "text/event-stream"
        return headers

    @staticmethod
    def _build_messages(
        messages: list[dict[str, Any]], image: bytes | None
    ) -> list[dict[str, Any]]:
        if image is None:
            return messages
        media_type = _media_type(image)
        encoded = base64.b64encode(image).decode("ascii")
        enriched = list(messages)
        last = dict(enriched[-1])
        last["content"] = [
            {"type": "text", "text": last.get("content", "")},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
            },
        ]
        enriched[-1] = last
        return enriched

    async def _send(
        self, headers: dict[str, str], payload: dict[str, Any], *, stream: bool
    ) -> httpx.Response:
        for attempt in range(2):
            try:
                if stream:
                    request = self._client.build_request(
                        "POST", CHAT_COMPLETIONS_PATH, json=payload, headers=headers
                    )
                    response = await self._client.send(request, stream=True)
                else:
                    response = await self._client.post(
                        CHAT_COMPLETIONS_PATH, json=payload, headers=headers
                    )
            except httpx.TimeoutException as exc:
                if attempt == 0:
                    await asyncio.sleep(self._retry_backoff_seconds)
                    continue
                raise DomainError(
                    code="LLM_TIMEOUT",
                    message="Muse Spark request timed out",
                    status_code=504,
                ) from exc

            if response.status_code in (401, 403):
                return response
            if attempt == 0 and (
                response.status_code in RETRYABLE_STATUS_CODES
                or response.status_code >= 500
            ):
                if stream:
                    await response.aclose()
                await asyncio.sleep(self._retry_backoff_seconds)
                continue
            return response

        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _raise_for_error(response: httpx.Response) -> None:
        if response.status_code in (401, 403):
            raise DomainError(
                code="LLM_AUTH_FAILED",
                message="Muse Spark rejected the API key",
                status_code=502,
            )
        if response.status_code >= 400:
            raise DomainError(
                code="LLM_REQUEST_FAILED",
                message=f"Muse Spark request failed with status {response.status_code}",
                status_code=502,
            )

    def _log_usage(self, usage: dict[str, Any] | None) -> None:
        if usage:
            logger.info("muse_spark_usage", extra={"usage": usage})
        if self._on_usage:
            self._on_usage(usage)


def _media_type(image: bytes) -> str:
    for magic, media_type in MEDIA_TYPES.items():
        if image.startswith(magic):
            return media_type
    raise ValueError("Only JPEG and PNG images can be sent to Muse Spark")


def _estimated_usage(
    messages: list[dict[str, Any]], image: bytes | None, streamed_characters: int
) -> dict[str, Any]:
    sent = sum(len(str(message.get("content", ""))) for message in messages)
    image_tokens = IMAGE_TOKEN_ESTIMATE if image else 0
    total = (sent + streamed_characters) // CHARACTERS_PER_TOKEN + image_tokens
    return {"total_tokens": total, "estimated": True}
