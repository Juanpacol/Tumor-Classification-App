"""Leakage guarantees of a training run and how each one is assured."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.training.dataset import Split
from src.training.folds import Splits

CHECKED = "checked in this run"
BY_DESIGN = "by design, unit-tested"


class LeakageError(ValueError):
    """A guarantee checked during the run does not hold."""


def leakage_guarantees(
    split: Split, splits: Splits, n_originals: int
) -> list[dict[str, Any]]:
    """The guarantees the report lists, with their status.

    ``splits`` are the splits the model is calibrated on, whose rows from
    ``n_originals`` on are augmented copies. G1 and G3 are checked on the data of
    this run and raise ``LeakageError`` when broken; the others hold by the
    structure of the code and are covered by unit tests.
    """
    shared = {s.group_id for s in split.train} & {s.group_id for s in split.test}
    if shared:
        raise LeakageError(f"group_id in both Train and Test: {sorted(shared)[:10]}")
    if any(np.any(validation >= n_originals) for _, validation in splits):
        raise LeakageError("an augmented copy is a validation row")
    return [
        {
            "id": "G1",
            "guarantee": "No group_id is in both Training and Testing",
            "status": CHECKED,
        },
        {
            "id": "G2",
            "guarantee": "Quality thresholds are fitted on Training images only",
            "status": BY_DESIGN,
        },
        {
            "id": "G3",
            "guarantee": "No augmented copy is a validation row",
            "status": CHECKED,
        },
        {
            "id": "G7",
            "guarantee": "Testing is scored exactly once",
            "status": BY_DESIGN,
        },
        {
            "id": "G8",
            "guarantee": "Scaler and PCA are fitted inside each fold",
            "status": BY_DESIGN,
        },
    ]
