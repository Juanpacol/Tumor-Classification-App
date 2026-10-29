import logging
from collections import deque
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from domain.errors import DomainError
from ports.llm_port import LLMPort

logger = logging.getLogger(__name__)

UNREPORTED_CALL_TOKENS = 4_000
IMAGE_TOKEN_ESTIMATE = 1_500
CHARACTERS_PER_TOKEN = 4
RATE_WINDOW = timedelta(minutes=1)


class SpendCap:
    """Daily token budget and per-minute request limit of one process.

    The counters live in memory: they reset when the process restarts and each
    instance of a scaled deployment counts on its own. The budget is checked
    before a call and the tokens are added after it, so one call can overshoot.
    """

    def __init__(
        self,
        daily_token_budget: int,
        max_requests_per_minute: int,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._daily_token_budget = daily_token_budget
        self._max_requests_per_minute = max_requests_per_minute
        self._clock = clock
        self._day = clock().date()
        self._tokens_today = 0
        self._recent_requests: deque[datetime] = deque()

    @property
    def tokens_today(self) -> int:
        self._roll_over_day()
        return self._tokens_today

    def check(self) -> None:
        """Count a request, or raise ``DomainError`` 429 when a limit is reached."""
        now = self._clock()
        self._roll_over_day()
        if self._tokens_today >= self._daily_token_budget:
            logger.warning(
                "llm_budget_exceeded",
                extra={
                    "tokens": self._tokens_today,
                    "budget": self._daily_token_budget,
                },
            )
            raise DomainError(
                "LLM_BUDGET_EXCEEDED",
                "The daily AI usage limit was reached. Try again tomorrow.",
                status_code=429,
            )
        while self._recent_requests and now - self._recent_requests[0] >= RATE_WINDOW:
            self._recent_requests.popleft()
        if len(self._recent_requests) >= self._max_requests_per_minute:
            logger.warning(
                "llm_rate_limited", extra={"limit": self._max_requests_per_minute}
            )
            raise DomainError(
                "LLM_RATE_LIMITED",
                "Too many AI requests. Try again in a minute.",
                status_code=429,
            )
        self._recent_requests.append(now)

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        """Add the tokens of a finished call; a call that reports none counts a default."""
        self.add_tokens(_total_tokens(usage) or UNREPORTED_CALL_TOKENS)

    def add_tokens(self, tokens: int) -> None:
        self._roll_over_day()
        self._tokens_today += tokens
        logger.info(
            "llm_tokens_counted",
            extra={"tokens": tokens, "today": self._tokens_today},
        )

    def _roll_over_day(self) -> None:
        today = self._clock().date()
        if today != self._day:
            self._day = today
            self._tokens_today = 0


class CappedLLM:
    """LLMPort that refuses calls once the spend cap is reached."""

    def __init__(self, inner: LLMPort, cap: SpendCap) -> None:
        self.inner = inner
        self._cap = cap

    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: type,
        image: bytes | None = None,
    ) -> Any:
        self._cap.check()
        return await self.inner.complete_json(messages, schema, image=image)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        image: bytes | None = None,
    ) -> AsyncIterator[str]:
        """Streamed answers report no usage, so their tokens are estimated.

        The limits are checked when iteration starts, not when ``stream`` is called.
        What was streamed is counted even if the reader stops early.
        """
        self._cap.check()
        characters = _characters(messages)
        try:
            async for chunk in self.inner.stream(messages, image=image):
                characters += len(chunk)
                yield chunk
        finally:
            image_tokens = IMAGE_TOKEN_ESTIMATE if image else 0
            self._cap.add_tokens(characters // CHARACTERS_PER_TOKEN + image_tokens)

    async def aclose(self) -> None:
        await self.inner.aclose()


def _total_tokens(usage: dict[str, Any] | None) -> int:
    if not usage:
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    pairs = (("prompt_tokens", "completion_tokens"), ("input_tokens", "output_tokens"))
    for prompt, completion in pairs:
        if isinstance(usage.get(prompt), int) or isinstance(usage.get(completion), int):
            return int(usage.get(prompt) or 0) + int(usage.get(completion) or 0)
    return 0


def _characters(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages)
