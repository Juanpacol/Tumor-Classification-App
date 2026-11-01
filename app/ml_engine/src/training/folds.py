"""Grouped fold assignment, made once per run and shared by every cross-validated step."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.training.dataset import Sample

N_SPLITS = 5

Splits = list[tuple[np.ndarray, np.ndarray]]


def assign_folds(
    samples: Sequence[Sample], n_splits: int, seed: int
) -> dict[Path, int]:
    """Fold of every sample, stratified by label and never splitting a ``group_id``.

    It uses only the manifest data of each sample, so it runs before feature
    extraction and the quality gate, and which images the gate keeps cannot move
    any image to another fold. The groups are near-duplicate clusters, the closest
    proxy for patients that the dataset offers.
    """
    labels = np.array([s.label.value for s in samples])
    groups = np.array([s.group_id for s in samples])
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of: dict[Path, int] = {}
    for fold, (_, validation) in enumerate(
        folds.split(np.zeros(len(samples)), labels, groups)
    ):
        for index in validation:
            fold_of[samples[index].path] = fold
    return fold_of


def fold_splits(
    fold_of_row: np.ndarray, n_splits: int, is_copy: np.ndarray | None = None
) -> Splits:
    """``(fit rows, validation rows)`` per fold, from the fold of each matrix row.

    Augmented copies (``is_copy``) carry their original's fold. They are fit rows of
    the other folds and never validation rows, so fold k validates exactly the
    original images of fold k. Raises ``ValueError`` naming the folds left without
    original rows. Without ``is_copy`` every row is an original.
    """
    originals = np.ones(len(fold_of_row), dtype=bool) if is_copy is None else ~is_copy
    empty = [
        fold
        for fold in range(n_splits)
        if not np.any((fold_of_row == fold) & originals)
    ]
    if empty:
        raise ValueError(f"folds without usable images: {empty}")
    return [
        (
            np.flatnonzero(fold_of_row != fold),
            np.flatnonzero((fold_of_row == fold) & originals),
        )
        for fold in range(n_splits)
    ]


@dataclass(frozen=True)
class FoldRow:
    """One Training image of the fold record; ``used`` is false when the gate rejected it."""

    path: str
    label: str
    group_id: int
    fold: int
    used: bool


def fold_record(
    samples: Sequence[Sample],
    fold_of: dict[Path, int],
    used: set[Path],
    data_dir: Path,
) -> list[FoldRow]:
    """Fold of every Training image, with paths relative to ``data_dir``."""
    return [
        FoldRow(
            s.path.relative_to(data_dir).as_posix(),
            s.label.value,
            s.group_id,
            fold_of[s.path],
            s.path in used,
        )
        for s in samples
    ]


def used_per_fold_and_class(
    record: Sequence[FoldRow], n_splits: int, labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Used images per fold (``"0"``..) and class, including zero counts."""
    counts = Counter((row.fold, row.label) for row in record if row.used)
    return {
        str(fold): {label: counts[(fold, label)] for label in labels}
        for fold in range(n_splits)
    }
