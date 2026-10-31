"""Backbone bundle: a ``state_dict`` file plus a manifest that describes it.

The file names are fixed, so no path is ever built from manifest content.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision

from src.deep_learning.backbones import get_spec
from src.errors import ModelUnavailableError

LICENSE_NOT_VERIFIED = "terms of use not verified"
WEIGHTS_FILE = "backbone.pt"
MANIFEST_FILE = "backbone.json"


@dataclass(frozen=True)
class BackboneManifest:
    """What the weights were exported from and what input they expect.

    ``source_url`` and ``license_note`` record the provenance of the weights, and
    have defaults so that bundles exported before they existed still load.
    """

    backbone: str
    embedding_dim: int
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    sha256: str
    torchvision_version: str
    weights_version: str
    source_url: str = ""
    license_note: str = LICENSE_NOT_VERIFIED


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
    *,
    input_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    source_url: str = "",
    license_note: str = LICENSE_NOT_VERIFIED,
) -> BackboneManifest:
    """Save the weights and a manifest with their sha256 and provenance into ``directory``.

    ``input_size``, ``mean`` and ``std`` are the input the weights were trained
    with; the CNN tensor is normalized with them (pipeline block B8).
    """
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, directory / WEIGHTS_FILE)
    manifest = BackboneManifest(
        backbone=backbone,
        embedding_dim=get_spec(backbone).embedding_dim,
        input_size=input_size,
        mean=tuple(mean),
        std=tuple(std),
        sha256=sha256_of(directory / WEIGHTS_FILE),
        torchvision_version=torchvision.__version__,
        weights_version=weights_version,
        source_url=source_url,
        license_note=license_note,
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
        _check_input(manifest)
    except (ValueError, KeyError, TypeError) as exc:
        raise ModelUnavailableError(f"invalid manifest {manifest_path}: {exc}") from exc

    if sha256_of(weights_path) != manifest.sha256:
        raise ModelUnavailableError("weights file does not match the manifest sha256")
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelUnavailableError(f"cannot load weights: {exc}") from exc
    return state_dict, manifest


def _check_input(manifest: BackboneManifest) -> None:
    """Raise ``ValueError`` unless the input size and normalization can be used."""
    if (
        not isinstance(manifest.input_size, int)
        or isinstance(manifest.input_size, bool)
        or manifest.input_size < 1
    ):
        raise ValueError(
            f"input_size must be a positive integer: {manifest.input_size}"
        )
    for name, values in (("mean", manifest.mean), ("std", manifest.std)):
        if len(values) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values
        ):
            raise ValueError(f"{name} must be 3 finite numbers: {values}")
    if min(manifest.std) <= 0:
        raise ValueError(f"std must be positive: {manifest.std}")
