from pathlib import Path

import numpy as np
import pytest

from src.contracts import TumorLabel
from src.training.dataset import Sample, Split
from src.training.guarantees import CHECKED, LeakageError, leakage_guarantees


def sample(split: str, group: int) -> Sample:
    return Sample(Path(f"{split}/{group}.png"), split, TumorLabel.GLIOMA, group)


CLEAN = Split([sample("Train", 1), sample("Train", 2)], [sample("Test", 3)], {})
SPLITS = [(np.array([1, 2]), np.array([0])), (np.array([0, 2]), np.array([1]))]


def test_a_clean_run_lists_every_guarantee_with_its_status() -> None:
    guarantees = leakage_guarantees(CLEAN, SPLITS, n_originals=2)

    assert [g["id"] for g in guarantees] == ["G1", "G2", "G3", "G7", "G8"]
    checked = {g["id"] for g in guarantees if g["status"] == CHECKED}
    assert checked == {"G1", "G3"}


def test_a_group_in_both_splits_stops_the_run() -> None:
    leaky = Split([sample("Train", 3)], [sample("Test", 3)], {})

    with pytest.raises(LeakageError, match="group_id in both"):
        leakage_guarantees(leaky, SPLITS, n_originals=2)


def test_a_copy_used_as_a_validation_row_stops_the_run() -> None:
    with pytest.raises(LeakageError, match="augmented copy"):
        leakage_guarantees(CLEAN, SPLITS, n_originals=1)
