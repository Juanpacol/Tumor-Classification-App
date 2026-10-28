class InMemoryStorage:
    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def save(self, data: bytes, key: str) -> str:
        self._store[key] = data
        return key

    def load(self, key: str) -> bytes:
        return self._store[key]
