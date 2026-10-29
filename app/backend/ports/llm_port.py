from collections.abc import AsyncIterator
from typing import Any, Protocol


class LLMPort(Protocol):
    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: type,
        image: bytes | None = None,
    ) -> Any: ...

    def stream(
        self,
        messages: list[dict[str, Any]],
        image: bytes | None = None,
    ) -> AsyncIterator[str]: ...

    async def aclose(self) -> None: ...
