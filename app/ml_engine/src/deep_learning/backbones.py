"""Registry of the pretrained backbones that can produce embeddings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn
from torchvision import models

DEFAULT_BACKBONE = "efficientnet_b0"


@dataclass(frozen=True)
class BackboneSpec:
    """How to build a backbone without weights and cut off its classification head."""

    name: str
    embedding_dim: int
    build: Callable[[], nn.Module]
    head_attribute: str


BACKBONES: dict[str, BackboneSpec] = {
    spec.name: spec
    for spec in (
        BackboneSpec(
            "efficientnet_b0",
            1280,
            lambda: models.efficientnet_b0(weights=None),
            "classifier",
        ),
        BackboneSpec("resnet50", 2048, lambda: models.resnet50(weights=None), "fc"),
        BackboneSpec(
            "densenet121", 1024, lambda: models.densenet121(weights=None), "classifier"
        ),
    )
}


def get_spec(name: str) -> BackboneSpec:
    """Look up a backbone by name, listing the valid names on failure."""
    try:
        return BACKBONES[name]
    except KeyError:
        raise ValueError(
            f"unknown backbone {name!r}, choose from {sorted(BACKBONES)}"
        ) from None


def build_backbone(name: str) -> nn.Module:
    """Architecture with the classification head intact, for loading a ``state_dict``."""
    return get_spec(name).build()


def as_feature_extractor(model: nn.Module, name: str) -> nn.Module:
    """Remove the head so the output is the pooled embedding, then freeze in eval mode."""
    setattr(model, get_spec(name).head_attribute, nn.Identity())
    model.eval()
    model.requires_grad_(False)
    return model
