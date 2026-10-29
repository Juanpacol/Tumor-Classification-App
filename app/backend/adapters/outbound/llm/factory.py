import logging

from adapters.outbound.llm.fake_llm import FakeLLM
from adapters.outbound.llm.meta_model_llm import MetaModelLLM
from adapters.outbound.llm.spend_cap import CappedLLM, SpendCap
from ports.llm_port import LLMPort
from settings import Settings

logger = logging.getLogger(__name__)


def build_llm(settings: Settings) -> LLMPort:
    """The LLM chosen by ``LLM_PROVIDER``.

    ``meta`` (the default) is the real adapter behind the spend cap. Without a
    key it still starts and every call answers 503 LLM_UNAVAILABLE, so a
    deployment never serves invented reports. ``fake`` gives canned answers, has
    no cost and is only for local development.
    """
    if settings.llm_provider == "fake":
        logger.warning("LLM provider is the fake: answers are canned")
        return FakeLLM()
    cap = SpendCap(
        settings.llm_daily_token_budget, settings.llm_max_requests_per_minute
    )
    logger.info(
        "LLM provider is Meta (%s, key configured: %s), capped at %d tokens a day "
        "and %d requests a minute",
        settings.meta_model_id,
        "yes" if settings.meta_model_api_key else "no",
        settings.llm_daily_token_budget,
        settings.llm_max_requests_per_minute,
    )
    meta = MetaModelLLM(
        api_key=settings.meta_model_api_key,
        model=settings.meta_model_id,
        on_usage=cap.record_usage,
    )
    return CappedLLM(meta, cap)
