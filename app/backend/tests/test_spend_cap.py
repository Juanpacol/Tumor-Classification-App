from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from adapters.outbound.llm.spend_cap import (
    IMAGE_TOKEN_ESTIMATE,
    UNREPORTED_CALL_TOKENS,
    CappedLLM,
    SpendCap,
)
from domain.errors import DomainError
from tests.doubles import ScriptedLLM


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


def cap(clock: Clock, tokens: int = 1_000, per_minute: int = 3) -> SpendCap:
    return SpendCap(tokens, per_minute, clock=clock)


def refusal(spend_cap: SpendCap) -> DomainError:
    with pytest.raises(DomainError) as error:
        spend_cap.check()
    return error.value


def test_requests_up_to_the_limit_pass_and_the_next_is_a_429() -> None:
    spend_cap = cap(Clock(), per_minute=3)
    for _ in range(3):
        spend_cap.check()

    error = refusal(spend_cap)

    assert error.status_code == 429 and error.code == "LLM_RATE_LIMITED"


def test_the_rate_window_slides() -> None:
    clock = Clock()
    spend_cap = cap(clock, per_minute=2)
    spend_cap.check()
    clock.advance(seconds=30)
    spend_cap.check()
    assert refusal(spend_cap).code == "LLM_RATE_LIMITED"

    clock.advance(seconds=30)
    spend_cap.check()

    assert refusal(spend_cap).code == "LLM_RATE_LIMITED"


def test_a_refused_request_does_not_count() -> None:
    clock = Clock()
    spend_cap = cap(clock, per_minute=1)
    spend_cap.check()
    for _ in range(5):
        refusal(spend_cap)

    clock.advance(seconds=60)

    spend_cap.check()


def test_the_daily_budget_blocks_once_it_is_reached() -> None:
    spend_cap = cap(Clock(), tokens=1_000, per_minute=100)
    spend_cap.record_usage({"total_tokens": 600})
    spend_cap.check()

    spend_cap.record_usage({"total_tokens": 400})
    error = refusal(spend_cap)

    assert error.status_code == 429 and error.code == "LLM_BUDGET_EXCEEDED"


def test_the_budget_resets_on_the_next_utc_day() -> None:
    clock = Clock()
    spend_cap = cap(clock, tokens=1_000, per_minute=100)
    spend_cap.record_usage({"total_tokens": 1_000})
    assert refusal(spend_cap).code == "LLM_BUDGET_EXCEEDED"

    clock.advance(hours=12)

    assert spend_cap.tokens_today == 0
    spend_cap.check()


def test_the_budget_is_reported_before_the_rate_limit() -> None:
    spend_cap = cap(Clock(), tokens=10, per_minute=1)
    spend_cap.check()
    spend_cap.add_tokens(10)

    assert refusal(spend_cap).code == "LLM_BUDGET_EXCEEDED"


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"total_tokens": 150, "prompt_tokens": 1, "completion_tokens": 1}, 150),
        ({"prompt_tokens": 120, "completion_tokens": 30}, 150),
        ({"input_tokens": 100, "output_tokens": 20}, 120),
        ({"completion_tokens": 30}, 30),
        (None, UNREPORTED_CALL_TOKENS),
        ({}, UNREPORTED_CALL_TOKENS),
        ({"something_else": 5}, UNREPORTED_CALL_TOKENS),
    ],
)
def test_usage_formats_and_calls_that_report_nothing(
    usage: dict[str, Any] | None, expected: int
) -> None:
    spend_cap = cap(Clock())

    spend_cap.record_usage(usage)

    assert spend_cap.tokens_today == expected


def test_limits_are_logged_when_hit(caplog: pytest.LogCaptureFixture) -> None:
    spend_cap = cap(Clock(), tokens=10)
    spend_cap.add_tokens(10)

    with caplog.at_level("WARNING"):
        refusal(spend_cap)

    assert "llm_budget_exceeded" in caplog.text


async def test_a_call_inside_the_limits_reaches_the_model() -> None:
    inner = ScriptedLLM("answer")
    llm = CappedLLM(inner, cap(Clock()))

    result = await llm.complete_json(
        [{"role": "user", "content": "q"}], dict, image=b"i"
    )

    assert result == "answer"
    assert inner.calls[0]["image"] == b"i"


async def test_a_call_over_the_limits_never_reaches_the_model() -> None:
    inner = ScriptedLLM("answer")
    spend_cap = cap(Clock(), per_minute=1)
    llm = CappedLLM(inner, spend_cap)
    await llm.complete_json([], dict)

    with pytest.raises(DomainError) as error:
        await llm.complete_json([], dict)

    assert error.value.status_code == 429 and len(inner.calls) == 1


class StreamingInner(ScriptedLLM):
    def __init__(self, *chunks: str) -> None:
        super().__init__()
        self._chunks = chunks

    async def stream(
        self,
        messages: list[dict[str, Any]],
        image: bytes | None = None,
    ) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


async def test_a_stream_is_passed_through_and_its_tokens_are_estimated() -> None:
    spend_cap = cap(Clock())
    llm = CappedLLM(StreamingInner("a" * 40, "b" * 40), spend_cap)

    chunks = [c async for c in llm.stream([{"role": "user", "content": "c" * 120}])]

    assert chunks == ["a" * 40, "b" * 40]
    assert spend_cap.tokens_today == (120 + 80) // 4


async def test_an_image_adds_its_estimate_to_a_stream() -> None:
    spend_cap = cap(Clock())
    llm = CappedLLM(StreamingInner("x" * 40), spend_cap)

    [c async for c in llm.stream([], image=b"png")]

    assert spend_cap.tokens_today == 40 // 4 + IMAGE_TOKEN_ESTIMATE


async def test_a_stream_the_reader_abandons_still_counts_what_was_sent() -> None:
    spend_cap = cap(Clock())
    llm = CappedLLM(StreamingInner("a" * 40, "b" * 40), spend_cap)

    stream = llm.stream([])
    await anext(stream)
    await stream.aclose()

    assert spend_cap.tokens_today == 40 // 4


async def test_a_stream_over_the_limits_fails_when_iteration_starts() -> None:
    spend_cap = cap(Clock(), per_minute=1)
    spend_cap.check()
    llm = CappedLLM(StreamingInner("x"), spend_cap)

    with pytest.raises(DomainError) as error:
        await anext(llm.stream([]))

    assert error.value.status_code == 429


async def test_closing_closes_the_wrapped_llm() -> None:
    class Closable(ScriptedLLM):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    inner = Closable()

    await CappedLLM(inner, cap(Clock())).aclose()

    assert inner.closed
