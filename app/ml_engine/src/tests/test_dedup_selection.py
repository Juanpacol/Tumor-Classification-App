from pathlib import Path

from src.preprocessing.deduplication.fingerprint import ImageFingerprint, compute_fingerprint
from src.preprocessing.deduplication.matching import find_similar_pairs, group_duplicates
from src.preprocessing.deduplication.selection import Decision, select_representatives
from src.tests.conftest import index_of
from src.tests.synthetic import save, synthetic_slice


def _decide(fingerprints: list[ImageFingerprint]) -> dict[str, Decision]:
    grouping = group_duplicates(fingerprints, find_similar_pairs(fingerprints, 6), 0.98, 0.95)
    decisions = select_representatives(fingerprints, grouping)
    return {fp.path.name: d for fp, d in zip(fingerprints, decisions, strict=True)}


def test_selection_rules(fingerprints: list[ImageFingerprint]) -> None:
    decisions = _decide(fingerprints)

    assert decisions["a.png"].keep != decisions["a_copy.png"].keep
    assert decisions["b.png"].keep, "larger resolution is kept"
    assert decisions["b_small.jpg"].keeper == index_of(fingerprints, "b.png")
    assert decisions["d.png"].keep
    assert decisions["Tr-aug-me_1.png"].reason == "augmented"
    assert decisions["e.png"].reason == decisions["e_other.png"].reason == "label_conflict"
    assert decisions["f.png"].keep


def test_group_size_counts_kept_images_only(fingerprints: list[ImageFingerprint]) -> None:
    decisions = _decide(fingerprints)

    assert decisions["b.png"].group_size == 1
    assert decisions["e.png"].group_size == 0


def test_label_conflict_keeps_majority_label(tmp_path: Path) -> None:
    image = synthetic_slice(7)
    paths = [
        save(image, tmp_path / "Train/notumor/n1.png"),
        save(image.resize((200, 200)), tmp_path / "Train/notumor/n2.png"),
        save(image, tmp_path / "Test/meningioma/m1.png"),
    ]
    decisions = _decide([compute_fingerprint(p, p.parts[-3], p.parts[-2]) for p in paths])

    assert decisions["n1.png"].reason == "kept"
    assert decisions["n2.png"].reason == "duplicate"
    assert decisions["m1.png"].reason == "label_conflict"
