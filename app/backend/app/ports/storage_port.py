from typing import Protocol


class StoragePort(Protocol):
    def save(self, data: bytes, key: str) -> str: ...

    def load(self, key: str) -> bytes: ...
