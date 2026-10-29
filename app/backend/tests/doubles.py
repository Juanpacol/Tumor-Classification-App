from collections.abc import AsyncIterator
from typing import Any


class ScriptedLLM:
    """LLM test double that answers with a queue of prepared responses.

    A response is a schema instance, a raw dict (validated like the real adapter
    does, so a wrong dict raises ``ValidationError``) or an exception to raise.
    """

    def __init__(self, *responses: Any) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: type,
        image: bytes | None = None,
    ) -> Any:
        self.calls.append({"messages": messages, "schema": schema, "image": image})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return (
            schema.model_validate(response) if isinstance(response, dict) else response
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        image: bytes | None = None,
    ) -> AsyncIterator[str]:
        for chunk in ():
            yield chunk

    async def aclose(self) -> None:
        return None
