"""Shared helpers for the tests that run the training pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.training.run import TrainingConfig

PER_CLASS = 20
NOW = datetime(2026, 10, 29, 10, 0, 0, tzinfo=UTC)


def make_config(
    workspace: dict[str, Path], out: Path, **overrides: Any
) -> TrainingConfig:
    return TrainingConfig(
        data_dir=workspace["data"],
        manifest=workspace["manifest"],
        backbone_bundle=workspace["backbone"],
        out_dir=out,
        n_splits=3,
        version="9.9.9",
        cache_dir=workspace["root"] / "cache",
        **overrides,
    )
