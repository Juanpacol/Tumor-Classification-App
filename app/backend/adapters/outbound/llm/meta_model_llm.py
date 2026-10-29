import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from domain.errors import DomainError

BASE_URL = "https://api.meta.ai/v1"
CHAT_COMPLETIONS_PATH = "/chat/completions"
CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 90.0
RETRY_BACKOFF_SECONDS = 0.5
RETRYABLE_STATUS_CODES = {429}

logger = logging.getLogger("adapters.outbound.llm.meta_model")


class MetaModelLLM:
    """LLMPort adapter for Muse Spark via the Meta Model API.

    See api.meta.ai/v1/chat/completions: Bearer auth, SSE streaming via
    `stream: true` with `choices[0].delta.content` chunks terminated by
    `data: [DONE]`, and structured output via `response_format.json_schema`.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str,
        client: httpx.AsyncClient | None = None,
        retry_backoff_seconds: float = RETRY_BACKOFF_SECONDS,
    ) -> None:
        self._api_key = api_key
        self._model = model
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
        api_key = self._require_api_key()
        payload = {
            "model": self._model,
            "messages": self._build_messages(messages, image),
            "stream": True,
        }
        response = await self._send(
            self._headers(api_key, streaming=True), payload, stream=True
        )
        self._raise_for_error(response)
        try:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield delta
        finally:
            await response.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

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
        encoded = base64.b64encode(image).decode("ascii")
        enriched = list(messages)
        last = dict(enriched[-1])
        last["content"] = [
            {"type": "text", "text": last.get("content", "")},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
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

    @staticmethod
    def _log_usage(usage: dict[str, Any] | None) -> None:
        if usage:
            logger.info("muse_spark_usage", extra={"usage": usage})
