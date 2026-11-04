"""The integration tests migrate and truncate their database, so they only run
against one named explicitly as disposable, never against DATABASE_URL."""

from collections.abc import Mapping

from sqlalchemy import make_url

TEST_DATABASE_URL = "TEST_DATABASE_URL"


class UnsafeTestDatabaseError(Exception):
    """TEST_DATABASE_URL names a database that does not look disposable."""


def disposable_database_url(environ: Mapping[str, str]) -> str | None:
    """Return TEST_DATABASE_URL, or None when unset.

    Raises UnsafeTestDatabaseError unless the database name contains "test",
    which keeps the compose dev database (ether_db) and Supabase out.
    """
    url = environ.get(TEST_DATABASE_URL)
    if not url:
        return None
    name = make_url(url).database or ""
    if "test" not in name.lower():
        raise UnsafeTestDatabaseError(
            f"{TEST_DATABASE_URL} names database {name!r}; the integration tests "
            "wipe it, so its name must contain 'test'"
        )
    return url
