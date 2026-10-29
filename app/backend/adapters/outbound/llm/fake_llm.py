from collections.abc import AsyncIterator
from typing import Any

CANNED_RESPONSES: dict[str, dict[str, Any]] = {
    "ClinicalReportOutput": {
        "findings": (
            "Demo report generated without a model. The result is uncertain and "
            "must not be used clinically."
        ),
        "visual_observations": "No image analysis was performed.",
        "impression": (
            "The finding is uncertain. Radiologist confirmation is required."
        ),
        "differential_considerations": "Not assessed in this demo.",
        "suggested_next_steps": ["Expert review of the study"],
        "ai_agreement": "uncertain",
    }
}


class FakeLLM:
    """LLMPort for local development without a Meta key. Its answers are canned."""

    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: type,
        image: bytes | None = None,
    ) -> Any:
        try:
            canned = CANNED_RESPONSES[schema.__name__]
        except KeyError:
            raise NotImplementedError(
                f"FakeLLM has no canned answer for {schema.__name__}"
            ) from None
        return schema.model_validate(canned)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        image: bytes | None = None,
    ) -> AsyncIterator[str]:
        for chunk in ("fake", " ", "response"):
            yield chunk

    async def aclose(self) -> None:
        return None
