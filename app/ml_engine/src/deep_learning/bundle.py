"""Backbone bundle: a ``state_dict`` file plus a manifest that describes it.

The file names are fixed, so no path is ever built from manifest content.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision

from src.deep_learning.backbones import get_spec
from src.errors import ModelUnavailableError

WEIGHTS_FILE = "backbone.pt"
MANIFEST_FILE = "backbone.json"


@dataclass(frozen=True)
class BackboneManifest:
    """What the weights were exported from and what input they expect."""

    backbone: str
    embedding_dim: int
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    sha256: str
    torchvision_version: str
    weights_version: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bundle(
    directory: Path,
    backbone: str,
    state_dict: dict[str, Any],
    weights_version: str,
    input_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> BackboneManifest:
    """Save the weights and a manifest with their sha256 into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, directory / WEIGHTS_FILE)
    manifest = BackboneManifest(
        backbone=backbone,
        embedding_dim=get_spec(backbone).embedding_dim,
        input_size=input_size,
        mean=mean,
        std=std,
        sha256=sha256_of(directory / WEIGHTS_FILE),
        torchvision_version=torchvision.__version__,
        weights_version=weights_version,
    )
    (directory / MANIFEST_FILE).write_text(json.dumps(asdict(manifest), indent=2))
    return manifest


def read_bundle(directory: Path) -> tuple[dict[str, Any], BackboneManifest]:
    """Read and verify a bundle; raise ``ModelUnavailableError`` when it cannot be used.

    The sha256 is checked before the weights are deserialized, and they are
    loaded with ``weights_only=True`` so a tampered file cannot run code.
    """
    weights_path = directory / WEIGHTS_FILE
    manifest_path = directory / MANIFEST_FILE
    for path in (weights_path, manifest_path):
        if not path.is_file():
            raise ModelUnavailableError(f"bundle file not found: {path}")

    try:
        raw = json.loads(manifest_path.read_text())
        manifest = BackboneManifest(
            **{**raw, "mean": tuple(raw["mean"]), "std": tuple(raw["std"])}
        )
        get_spec(manifest.backbone)
    except (ValueError, KeyError, TypeError) as exc:
        raise ModelUnavailableError(f"invalid manifest {manifest_path}: {exc}") from exc

    if sha256_of(weights_path) != manifest.sha256:
        raise ModelUnavailableError("weights file does not match the manifest sha256")
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelUnavailableError(f"cannot load weights: {exc}") from exc
    return state_dict, manifest
