import pytest

from tests.integration.database_guard import (
    UnsafeTestDatabaseError,
    disposable_database_url,
)

TEST_DB = "postgresql://test_user:test_password@postgres-test:5432/test_db"


def test_without_a_test_database_url_there_is_none_even_with_database_url() -> None:
    environ = {"DATABASE_URL": TEST_DB}

    assert disposable_database_url(environ) is None


def test_a_database_named_for_tests_is_used() -> None:
    assert disposable_database_url({"TEST_DATABASE_URL": TEST_DB}) == TEST_DB


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://ether_user:ether_password@postgres:5432/ether_db",
        "postgresql://user:password@pooler.supabase.com:5432/postgres",
        "postgresql://user:password@localhost:5432",
    ],
)
def test_any_other_database_is_refused(url: str) -> None:
    with pytest.raises(UnsafeTestDatabaseError, match="must contain 'test'"):
        disposable_database_url({"TEST_DATABASE_URL": url})
