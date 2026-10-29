from adapters.outbound.persistence.sqlalchemy.engine import connect_args, create_engine
from settings import Settings

URL = "postgresql://user:pw@host:5432/db"


def test_statement_cache_left_to_driver_default_when_unset():
    assert connect_args(Settings(database_url=URL, db_statement_cache_size=None)) == {}


def test_statement_cache_disabled_for_pooler():
    settings = Settings(database_url=URL, db_statement_cache_size=0)

    assert connect_args(settings) == {"statement_cache_size": 0}


def test_engine_uses_asyncpg_dialect():
    engine = create_engine(Settings(database_url=URL))

    assert engine.dialect.driver == "asyncpg"
