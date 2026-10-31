from pathlib import Path


class FilesystemStorage:
    """StoragePort backed by a directory on disk.

    Not persistent across container restarts or scale-to-zero on Cloud Run -
    see docs/deployment.md ("What's simplified for the cloud demo").
    """

    def __init__(self, root: str) -> None:
        self._root = Path(root)

    def save(self, data: bytes, key: str) -> str:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def load(self, key: str) -> bytes:
        path = self._root / key
        if not path.is_file():
            raise KeyError(key)
        return path.read_bytes()
