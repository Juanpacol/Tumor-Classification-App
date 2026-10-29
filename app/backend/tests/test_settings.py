import pytest

from settings import Settings


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://user:pw@host:5432/db",
        "postgres://user:pw@host:5432/db",
        "postgresql+asyncpg://user:pw@host:5432/db",
    ],
)
def test_async_database_url_uses_asyncpg_dialect(url: str):
    settings = Settings(database_url=url)

    assert settings.async_database_url == "postgresql+asyncpg://user:pw@host:5432/db"


def test_empty_statement_cache_size_falls_back_to_driver_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DB_STATEMENT_CACHE_SIZE", "")

    assert Settings(_env_file=None).db_statement_cache_size is None  # type: ignore[call-arg]


def test_default_database_is_postgres_not_sqlite(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.async_database_url.startswith("postgresql+asyncpg://")
