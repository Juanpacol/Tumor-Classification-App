import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from adapters.outbound.persistence.sqlalchemy.engine import create_engine
from adapters.outbound.persistence.sqlalchemy.orm import Base
from settings import get_settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata
MIGRATION_LOCK_ID = 218_000_001


def run_migrations_offline() -> None:
    context.configure(
        url=get_settings().async_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        # Serializes concurrent upgrades (e.g. compose up plus a one-off run);
        # the lock is released when the transaction ends.
        connection.exec_driver_sql(f"SELECT pg_advisory_xact_lock({MIGRATION_LOCK_ID})")
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_engine(get_settings())
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
