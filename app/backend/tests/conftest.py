from collections.abc import Iterator

import pytest

from settings import get_settings

# Set by the compose backend; each test that needs one sets it itself.
DEPLOYMENT_ENV = ("LLM_PROVIDER", "META_MODEL_API_KEY", "MODEL_BUNDLE_PATH")


@pytest.fixture(autouse=True)
def isolated_from_deployment_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in DEPLOYMENT_ENV:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
