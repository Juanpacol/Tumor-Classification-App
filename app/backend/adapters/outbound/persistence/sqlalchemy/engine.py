from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from settings import Settings


def connect_args(settings: Settings) -> dict[str, Any]:
    """Driver arguments; disables asyncpg's statement cache for PgBouncer poolers."""
    if settings.db_statement_cache_size is None:
        return {}
    return {"statement_cache_size": settings.db_statement_cache_size}


def create_engine(settings: Settings) -> AsyncEngine:
    """Build the async engine for the configured PostgreSQL database."""
    return create_async_engine(
        settings.async_database_url,
        pool_pre_ping=True,
        connect_args=connect_args(settings),
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory whose objects stay usable after commit."""
    return async_sessionmaker(engine, expire_on_commit=False)
