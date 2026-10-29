"""Locate dataset images by split and label folder."""

from __future__ import annotations

from pathlib import Path

SPLITS = ("Train", "Test")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def discover_images(data_dir: Path) -> list[tuple[Path, str, str]]:
    """List ``(path, split, label)`` for every image under ``data_dir/<split>/<label>/``."""
    entries: list[tuple[Path, str, str]] = []
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for path in sorted(label_dir.iterdir()):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    entries.append((path, split, label_dir.name))
    return entries
