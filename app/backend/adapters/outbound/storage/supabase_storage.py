import httpx

_NOT_FOUND = {400, 404}
_PNG_SIGNATURE = b"\x89PNG"


def _content_type(data: bytes) -> str:
    return "image/png" if data.startswith(_PNG_SIGNATURE) else "image/jpeg"


class SupabaseStorage:
    """StoragePort backed by a private Supabase Storage bucket.

    Survives Cloud Run restarts and scale-to-zero, unlike FilesystemStorage.
    The service key is a secret: it only reaches the backend as an environment
    variable and is never sent to the browser.
    """

    def __init__(
        self,
        url: str,
        service_key: str,
        bucket: str,
        client: httpx.Client | None = None,
    ) -> None:
        self._base = f"{url.rstrip('/')}/storage/v1/object/{bucket}"
        self._client = client or httpx.Client(timeout=30.0)
        self._headers = {
            "Authorization": f"Bearer {service_key}",
            "apikey": service_key,
        }

    def save(self, data: bytes, key: str) -> str:
        response = self._client.post(
            f"{self._base}/{key}",
            content=data,
            headers={
                **self._headers,
                "Content-Type": _content_type(data),
                "x-upsert": "true",
            },
        )
        response.raise_for_status()
        return key

    def load(self, key: str) -> bytes:
        response = self._client.get(f"{self._base}/{key}", headers=self._headers)
        if response.status_code in _NOT_FOUND:
            raise KeyError(key)
        response.raise_for_status()
        return response.content
