"""The leakage-free split on the cleaning manifest versioned in the repository."""

from pathlib import Path

from src.contracts import TumorLabel
from src.training.dataset import load_samples, split_samples

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MANIFEST = DATA_DIR / "processed" / "dedup_20261028T234450903626Z" / "manifest.csv"


def test_versioned_manifest_purge_matches_the_reference_run() -> None:
    split = split_samples(load_samples(DATA_DIR, MANIFEST))

    assert split.removed_per_class == {
        TumorLabel.GLIOMA: 142,
        TumorLabel.MENINGIOMA: 93,
        TumorLabel.NO_TUMOR: 15,
        TumorLabel.PITUITARY: 245,
    }
    assert len(split.train) == 1830
    assert len(split.test) == 1109
    assert not {s.group_id for s in split.train} & {s.group_id for s in split.test}
