import logging

from adapters.outbound.llm.fake_llm import FakeLLM
from adapters.outbound.llm.meta_model_llm import MetaModelLLM
from ports.llm_port import LLMPort
from settings import Settings

logger = logging.getLogger(__name__)


def build_llm(settings: Settings) -> LLMPort:
    """The LLM chosen by ``LLM_PROVIDER``.

    ``meta`` (the default) is the real adapter. Without a key it still starts and
    every call answers 503 LLM_UNAVAILABLE, so a deployment never serves invented
    reports. ``fake`` gives canned answers and is only for local development.
    """
    if settings.llm_provider == "fake":
        logger.warning("LLM provider is the fake: answers are canned")
        return FakeLLM()
    logger.info(
        "LLM provider is Meta (%s, key configured: %s)",
        settings.meta_model_id,
        "yes" if settings.meta_model_api_key else "no",
    )
    return MetaModelLLM(
        api_key=settings.meta_model_api_key, model=settings.meta_model_id
    )
