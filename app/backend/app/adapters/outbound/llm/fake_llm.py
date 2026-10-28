from collections.abc import AsyncIterator
from typing import Any


class FakeLLM:
    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: type,
        image: bytes | None = None,
    ) -> dict[str, Any]:
        return {}

    async def stream(
        self,
        messages: list[dict[str, Any]],
        image: bytes | None = None,
    ) -> AsyncIterator[str]:
        for chunk in ("fake", " ", "response"):
            yield chunk
