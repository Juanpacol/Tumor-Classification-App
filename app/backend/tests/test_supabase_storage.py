import httpx
import pytest

from adapters.outbound.storage.factory import build_storage
from adapters.outbound.storage.filesystem_storage import FilesystemStorage
from adapters.outbound.storage.supabase_storage import SupabaseStorage
from settings import Settings

URL = "https://project.supabase.co"


def _storage(handler) -> SupabaseStorage:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return SupabaseStorage(URL, "service-key", "study-images", client=client)


def test_save_uploads_the_bytes_with_the_service_key_and_returns_the_key():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"Key": "study-images/a/b.png"})

    assert _storage(handler).save(b"png-bytes", "a/b.png") == "a/b.png"

    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == f"{URL}/storage/v1/object/study-images/a/b.png"
    assert request.content == b"png-bytes"
    assert request.headers["content-type"] == "image/jpeg"
    assert request.headers["authorization"] == "Bearer service-key"
    assert request.headers["x-upsert"] == "true"


def test_save_declares_png_by_its_signature():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    _storage(handler).save(b"\x89PNG\r\n", "a.png")

    assert seen[0].headers["content-type"] == "image/png"


def test_load_returns_the_stored_bytes():
    storage = _storage(lambda request: httpx.Response(200, content=b"png-bytes"))

    assert storage.load("a/b.png") == b"png-bytes"


@pytest.mark.parametrize("status", [400, 404])
def test_load_of_a_missing_object_raises_key_error(status: int):
    storage = _storage(lambda request: httpx.Response(status, json={"error": "x"}))

    with pytest.raises(KeyError):
        storage.load("missing.png")


def test_save_surfaces_a_storage_failure():
    storage = _storage(lambda request: httpx.Response(500))

    with pytest.raises(httpx.HTTPStatusError):
        storage.save(b"x", "a.png")


def test_the_local_disk_is_used_when_supabase_is_not_configured():
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert isinstance(build_storage(settings), FilesystemStorage)


def test_supabase_is_used_when_its_url_and_key_are_set():
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, supabase_url=URL, supabase_service_key="service-key"
    )

    assert isinstance(build_storage(settings), SupabaseStorage)


def test_the_local_disk_is_kept_when_only_the_url_is_set():
    settings = Settings(_env_file=None, supabase_url=URL)  # type: ignore[call-arg]

    assert isinstance(build_storage(settings), FilesystemStorage)
