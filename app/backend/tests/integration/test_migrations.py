import asyncio

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from adapters.outbound.persistence.sqlalchemy.engine import create_engine
from settings import Settings

AGGREGATE_TABLES = {"studies", "predictions", "reports", "chat_messages"}
PHI_MARKERS = ("patient", "birth", "dob", "filename", "file_name")


async def test_database_is_migrated_to_head(engine: AsyncEngine, alembic_config: Config):
    head = ScriptDirectory.from_config(alembic_config).get_current_head()

    async with engine.connect() as connection:
        current = await connection.scalar(text("SELECT version_num FROM alembic_version"))

    assert current == head


async def _public_columns(settings: Settings) -> dict[str, set[str]]:
    engine = create_engine(settings)
    try:
        async with engine.connect() as connection:
            rows = await connection.execute(
                text(
                    "SELECT table_name, column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name <> 'alembic_version'"
                )
            )
            columns: dict[str, set[str]] = {}
            for table, column in rows:
                columns.setdefault(table, set()).add(column)
            return columns
    finally:
        await engine.dispose()


def test_migrated_schema_has_no_phi_columns(migrated_database: Settings):
    columns = asyncio.run(_public_columns(migrated_database))

    assert set(columns) == AGGREGATE_TABLES
    offending = [
        f"{table}.{column}"
        for table, names in columns.items()
        for column in names
        if any(marker in column for marker in PHI_MARKERS)
    ]
    assert not offending, f"PHI-like columns found: {offending}"


def test_initial_migration_downgrades_and_upgrades(
    migrated_database: Settings, alembic_config: Config
):
    try:
        command.downgrade(alembic_config, "base")
        assert asyncio.run(_public_columns(migrated_database)) == {}
    finally:
        command.upgrade(alembic_config, "head")

    assert set(asyncio.run(_public_columns(migrated_database))) == AGGREGATE_TABLES
