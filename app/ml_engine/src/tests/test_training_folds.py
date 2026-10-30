from pathlib import Path

import numpy as np
import pytest

from src.contracts import TumorLabel
from src.training.dataset import Sample
from src.training.folds import assign_folds, fold_splits

LABELS = list(TumorLabel)


def samples(n_per_class: int = 20, group_size: int = 2) -> list[Sample]:
    """Four classes; every ``group_size`` consecutive images of a class share a group."""
    result = []
    for class_index, label in enumerate(LABELS):
        for i in range(n_per_class):
            group = class_index * 1000 + i // group_size
            result.append(Sample(Path(f"{label.value}/{i}.png"), "Train", label, group))
    return result


def test_every_sample_gets_exactly_one_fold() -> None:
    data = samples()

    fold_of = assign_folds(data, 5, 0)

    assert set(fold_of) == {s.path for s in data}
    assert set(fold_of.values()) == set(range(5))


def test_a_group_never_straddles_two_folds() -> None:
    data = samples(group_size=3)

    fold_of = assign_folds(data, 5, 0)

    folds_of_group: dict[int, set[int]] = {}
    for s in data:
        folds_of_group.setdefault(s.group_id, set()).add(fold_of[s.path])
    assert all(len(folds) == 1 for folds in folds_of_group.values())


def test_folds_are_stratified() -> None:
    fold_of = assign_folds(samples(), 5, 0)

    for fold in range(5):
        per_class = [
            sum(1 for s in samples() if s.label == label and fold_of[s.path] == fold)
            for label in LABELS
        ]
        assert min(per_class) >= 2


def test_the_same_seed_gives_the_same_assignment() -> None:
    assert assign_folds(samples(), 5, 7) == assign_folds(samples(), 5, 7)


def test_a_different_seed_changes_the_assignment() -> None:
    assert assign_folds(samples(), 5, 1) != assign_folds(samples(), 5, 2)


def test_splits_use_the_fold_of_each_row() -> None:
    fold_of_row = np.array([0, 1, 2, 0, 1, 2])

    splits = fold_splits(fold_of_row, 3)

    assert [v.tolist() for _, v in splits] == [[0, 3], [1, 4], [2, 5]]
    assert [f.tolist() for f, _ in splits] == [[1, 2, 4, 5], [0, 2, 3, 5], [0, 1, 3, 4]]


def test_rows_dropped_by_the_gate_do_not_move_the_others() -> None:
    data = samples()
    fold_of = assign_folds(data, 5, 0)
    kept = data[::3]

    splits = fold_splits(np.array([fold_of[s.path] for s in kept]), 5)

    for fold, (_, validation) in enumerate(splits):
        assert all(fold_of[kept[i].path] == fold for i in validation)


def test_an_empty_fold_is_an_error_that_names_it() -> None:
    with pytest.raises(ValueError, match=r"\[1\]"):
        fold_splits(np.array([0, 0, 2]), 3)


def test_fewer_than_two_folds_is_rejected() -> None:
    with pytest.raises(ValueError):
        assign_folds(samples(), 1, 0)
