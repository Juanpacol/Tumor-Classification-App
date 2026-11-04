"""Pack ``model-v<version>/`` into one archive with its checksum and release notes."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from src.deep_learning.bundle import sha256_of
from src.errors import ModelUnavailableError
from src.training.artifact import MANIFEST_FILE

TAG_PREFIX = "model-v"


def archive_name(version: str) -> str:
    return f"{TAG_PREFIX}{version}.tar.gz"


def package(bundle_dir: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    """Write the archive, its ``.sha256`` file and the release notes to ``out_dir``."""
    manifest_path = bundle_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ModelUnavailableError(f"no {MANIFEST_FILE} in {bundle_dir}")
    manifest = json.loads(manifest_path.read_text())
    version = str(manifest["version"])
    if bundle_dir.name != f"{TAG_PREFIX}{version}":
        raise ModelUnavailableError(
            f"folder {bundle_dir.name} does not match manifest version {version}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / archive_name(version)
    if archive.exists():
        raise ModelUnavailableError(f"{archive} already exists")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)
    checksum = archive.with_name(archive.name + ".sha256")
    checksum.write_text(f"{sha256_of(archive)}  {archive.name}\n")
    notes = out_dir / f"release-notes-{TAG_PREFIX}{version}.md"
    notes.write_text(release_notes(manifest))
    return archive, checksum, notes


def release_notes(manifest: dict) -> str:
    """Release notes with the test metrics recorded in the manifest."""
    test = manifest["metrics"]["test"]
    trained = (
        f"Trained {manifest['created_at']} with seed {manifest['seed']}; "
        f"classifier `{manifest['classifier']}`."
    )
    lines = [
        f"# Model {manifest['version']}",
        "",
        trained,
        "",
        "## Test metrics",
        "",
        f"- accuracy: {test['accuracy']:.3f}",
        f"- macro F1: {test['macro_f1']:.3f}",
        f"- balanced accuracy: {test['balanced_accuracy']:.3f}",
        "",
        "| class | precision | recall | F1 | support |",
        "|---|---|---|---|---|",
    ]
    for label, row in test["per_class"].items():
        lines.append(
            f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} "
            f"| {row['f1']:.3f} | {row['support']} |"
        )
    lines += ["", "Trained and scored on public sample data. Not for clinical use.", ""]
    return "\n".join(lines)
