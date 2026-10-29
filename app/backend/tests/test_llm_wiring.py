import logging
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from adapters.outbound.llm import factory
from adapters.outbound.llm.factory import build_llm
from adapters.outbound.llm.fake_llm import FakeLLM
from adapters.outbound.llm.meta_model_llm import MetaModelLLM
from adapters.outbound.persistence.in_memory_repository import InMemoryStudyRepository
from application.clinical_report import ClinicalReportOutput
from application.report_guardrails import find_violations
from domain.errors import DomainError
from main import create_app
from settings import Settings, get_settings
from tests.factories import make_prediction


@pytest.fixture()
def fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def factory_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    # Alembic's fileConfig, run by the integration tests, disables existing loggers
    monkeypatch.setattr(factory.logger, "disabled", False)


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_the_default_provider_is_the_real_one() -> None:
    assert settings().llm_provider == "meta"


def test_an_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        settings(llm_provider="llama")


def test_with_a_key_the_real_adapter_is_built() -> None:
    llm = build_llm(settings(meta_model_api_key="a-key"))

    assert isinstance(llm, MetaModelLLM)


async def test_without_a_key_the_real_adapter_answers_503_instead_of_inventing() -> (
    None
):
    llm = build_llm(settings(meta_model_api_key=None))

    assert isinstance(llm, MetaModelLLM)
    with pytest.raises(DomainError) as error:
        await llm.complete_json([], ClinicalReportOutput)

    assert error.value.status_code == 503 and error.value.code == "LLM_UNAVAILABLE"


def test_the_fake_is_only_built_when_asked_for() -> None:
    assert isinstance(build_llm(settings(llm_provider="fake")), FakeLLM)


def test_the_choice_is_logged_without_the_key(
    factory_logging: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        build_llm(settings(meta_model_api_key="super-secret-key"))
        build_llm(settings(llm_provider="fake"))

    assert "key configured: yes" in caplog.text
    assert "fake" in caplog.text
    assert "super-secret-key" not in caplog.text


def test_the_app_starts_with_the_real_adapter_even_without_a_key(
    fresh_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("META_MODEL_API_KEY", raising=False)
    app = create_app(repository=InMemoryStudyRepository())

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert isinstance(app.state.llm, MetaModelLLM)


def test_the_app_starts_with_the_fake_when_configured(
    fresh_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    app = create_app(repository=InMemoryStudyRepository())

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert isinstance(app.state.llm, FakeLLM)


def test_the_llm_is_closed_when_the_app_shuts_down() -> None:
    class SpyLLM(FakeLLM):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    spy = SpyLLM()
    app = create_app(repository=InMemoryStudyRepository(), llm=spy)

    with TestClient(app):
        assert spy.closed is False

    assert spy.closed is True


async def test_the_real_adapter_closes_its_http_client() -> None:
    client = httpx.AsyncClient()
    llm = MetaModelLLM(api_key="a-key", model="muse-spark-1.3", client=client)

    await llm.aclose()

    assert client.is_closed


@pytest.mark.parametrize("triage", [True, False])
async def test_the_fake_report_is_valid_and_passes_the_guardrails(triage: bool) -> None:
    report = await FakeLLM().complete_json([], ClinicalReportOutput)

    assert isinstance(report, ClinicalReportOutput)
    prediction = make_prediction("s", needs_expert_triage=triage)
    assert find_violations(report, prediction) == []


async def test_the_fake_has_no_answer_for_an_unknown_schema() -> None:
    class Other(ClinicalReportOutput):
        pass

    with pytest.raises(NotImplementedError, match="Other"):
        await FakeLLM().complete_json([], Other)


async def test_the_fake_streams_canned_chunks() -> None:
    chunks = [chunk async for chunk in FakeLLM().stream([])]

    assert "".join(chunks) == "fake response"
