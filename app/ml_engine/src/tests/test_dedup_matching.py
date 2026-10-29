import pytest

from src.preprocessing.deduplication.fingerprint import ImageFingerprint
from src.preprocessing.deduplication.matching import (
    SimilarPair,
    find_similar_pairs,
    group_duplicates,
)
from src.tests.conftest import index_of


def test_duplicates_are_clustered(fingerprints: list[ImageFingerprint]) -> None:
    grouping = group_duplicates(fingerprints, find_similar_pairs(fingerprints, 6), 0.98, 0.95)

    def same(x: str, y: str) -> bool:
        ids = grouping.duplicate_ids
        return ids[index_of(fingerprints, x)] == ids[index_of(fingerprints, y)]

    assert same("a.png", "a_copy.png")
    assert same("b.png", "b_small.jpg")
    assert same("c.png", "c_flip.png")
    assert same("d.png", "Tr-aug-me_1.png")
    assert not same("a.png", "b.png")


def test_candidate_without_correlation_is_not_merged(fingerprints: list[ImageFingerprint]) -> None:
    a, b = index_of(fingerprints, "a.png"), index_of(fingerprints, "b.png")
    pairs = find_similar_pairs(fingerprints, max_distance=64)

    grouping = group_duplicates(fingerprints, pairs, 0.98, 0.95)

    assert any({p.first, p.second} == {a, b} for p in pairs)
    assert grouping.group_ids[a] != grouping.group_ids[b]


def test_related_pairs_share_group_only_within_label(fingerprints: list[ImageFingerprint]) -> None:
    b, c, f = (index_of(fingerprints, n) for n in ("b.png", "c.png", "f.png"))
    pairs = [SimilarPair(b, c, 0.96), SimilarPair(b, f, 0.97)]

    grouping = group_duplicates(fingerprints, pairs, 0.98, 0.95)

    assert grouping.group_ids[b] == grouping.group_ids[c]
    assert grouping.duplicate_ids[b] != grouping.duplicate_ids[c]
    assert grouping.group_ids[b] != grouping.group_ids[f]


def test_augmented_images_do_not_bridge_groups(fingerprints: list[ImageFingerprint]) -> None:
    aug, d, e = (index_of(fingerprints, n) for n in ("Tr-aug-me_1.png", "d.png", "e.png"))
    pairs = [SimilarPair(aug, d, 0.96), SimilarPair(aug, e, 0.96)]

    grouping = group_duplicates(fingerprints, pairs, 0.98, 0.95)

    assert grouping.group_ids[d] != grouping.group_ids[e]


def test_rejects_related_above_duplicate_threshold(fingerprints: list[ImageFingerprint]) -> None:
    with pytest.raises(ValueError):
        group_duplicates(fingerprints, [], 0.95, 0.99)
