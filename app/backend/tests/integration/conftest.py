import asyncio
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
from settings import Settings

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
    settings = Settings()
    if not asyncio.run(_is_reachable(settings)):
        url = make_url(settings.database_url).render_as_string(hide_password=True)
        pytest.skip(f"PostgreSQL not reachable at DATABASE_URL ({url})")
    command.upgrade(alembic_config, "head")
    yield settings


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
