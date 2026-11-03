"""Shared helpers for the tests that run the training pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.training.config import TrainingConfig

PER_CLASS = 20
NOW = datetime(2026, 10, 29, 10, 0, 0, tzinfo=UTC)


def make_config(
    workspace: dict[str, Path], out: Path, **overrides: Any
) -> TrainingConfig:
    """A final run on the synthetic workspace unless ``final=False`` is passed."""
    values = {
        "n_splits": 3,
        "version": "9.9.9",
        "cache_dir": workspace["root"] / "cache",
        "weights_dir": workspace["weights"],
        "final": True,
        **overrides,
    }
    return TrainingConfig(
        data_dir=workspace["data"],
        manifest=workspace["manifest"],
        out_dir=out,
        **values,
    )
