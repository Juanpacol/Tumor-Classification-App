import asyncio
import os
import pathlib
from collections.abc import AsyncIterator, Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import make_url, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from adapters.outbound.persistence.sqlalchemy.engine import (
    create_engine,
    create_session_factory,
)
from adapters.outbound.persistence.sqlalchemy.repository import (
    SqlAlchemyStudyRepository,
)
from settings import Settings, get_settings
from tests.integration.database_guard import (
    TEST_DATABASE_URL,
    disposable_database_url,
)

ALEMBIC_INI = pathlib.Path(__file__).resolve().parents[2] / "alembic.ini"
TABLES = "chat_messages, reports, predictions, studies"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    here = pathlib.Path(__file__).parent
    for item in items:
        if here in item.path.parents:
            item.add_marker(pytest.mark.integration)


async def _is_reachable(settings: Settings) -> bool:
    engine = create_engine(settings)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except (SQLAlchemyError, OSError):
        return False
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def alembic_config() -> Config:
    return Config(str(ALEMBIC_INI))


@pytest.fixture(scope="session")
def migrated_database(alembic_config: Config) -> Iterator[Settings]:
    url = disposable_database_url(os.environ)
    if url is None:
        pytest.skip(f"{TEST_DATABASE_URL} not set (DATABASE_URL is never used)")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DATABASE_URL", url)
        get_settings.cache_clear()
        settings = Settings()
        if not asyncio.run(_is_reachable(settings)):
            shown = make_url(url).render_as_string(hide_password=True)
            pytest.skip(f"PostgreSQL not reachable at {TEST_DATABASE_URL} ({shown})")
        command.upgrade(alembic_config, "head")
        yield settings
    get_settings.cache_clear()


@pytest.fixture(scope="session")
async def engine(migrated_database: Settings) -> AsyncIterator[AsyncEngine]:
    engine = create_engine(migrated_database)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    async with engine.begin() as connection:
        await connection.execute(text(f"TRUNCATE {TABLES} CASCADE"))
    return create_session_factory(engine)


@pytest.fixture
def repository(
    session_factory: async_sessionmaker[AsyncSession],
) -> SqlAlchemyStudyRepository:
    return SqlAlchemyStudyRepository(session_factory)
