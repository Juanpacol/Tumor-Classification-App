"""Download a published model bundle and check it against its SHA-256."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

from src.deep_learning.bundle import sha256_of
from src.errors import ModelUnavailableError
from src.release.package import TAG_PREFIX, archive_name

RELEASE_URL = "https://github.com/{repo}/releases/download/{tag}/{name}"


def fetch(version: str, repo: str, dest: Path) -> Path:
    """Fetch ``model-v<version>`` into ``dest`` and return the bundle folder.

    Nothing is extracted when the checksum does not match.
    """
    tag = f"{TAG_PREFIX}{version}"
    name = archive_name(version)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / name
    _download(RELEASE_URL.format(repo=repo, tag=tag, name=name), archive)
    expected = _expected_digest(
        RELEASE_URL.format(repo=repo, tag=tag, name=name + ".sha256")
    )
    verify(archive, expected)
    return extract(archive, dest, tag)


def verify(archive: Path, expected: str) -> None:
    actual = sha256_of(archive)
    if actual != expected:
        raise ModelUnavailableError(
            f"checksum mismatch for {archive.name}: expected {expected}, got {actual}"
        )


def extract(archive: Path, dest: Path, tag: str) -> Path:
    """Extract only regular files under ``<tag>/``; anything else aborts."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != tag
            ):
                raise ModelUnavailableError(
                    f"unexpected path in archive: {member.name}"
                )
            if not (member.isfile() or member.isdir()):
                raise ModelUnavailableError(
                    f"unexpected entry in archive: {member.name}"
                )
        tar.extractall(dest, members=members)
    return dest / tag


def _download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)


def _expected_digest(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode().split()[0]
