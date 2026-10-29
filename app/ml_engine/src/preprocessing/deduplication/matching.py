"""Find similar image pairs and cluster them into duplicates and related groups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.preprocessing.deduplication.fingerprint import ImageFingerprint, thumbnail_correlation
from src.preprocessing.deduplication.hashing import hamming_distances


@dataclass(frozen=True)
class SimilarPair:
    """Two pHash candidates and their thumbnail correlation."""

    first: int
    second: int
    correlation: float


@dataclass(frozen=True)
class Grouping:
    """Per-image cluster ids: duplicates (same slice) and related groups (for splitting)."""

    duplicate_ids: list[int]
    group_ids: list[int]


class UnionFind:
    """Disjoint-set structure with path compression."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        """Return the root of ``item``'s set."""
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        """Merge the sets containing ``first`` and ``second``."""
        self._parent[self.find(second)] = self.find(first)

    def component_ids(self) -> list[int]:
        """Consecutive component ids (from 0) in order of first appearance."""
        roots = [self.find(index) for index in range(len(self._parent))]
        ids = {root: new_id for new_id, root in enumerate(dict.fromkeys(roots))}
        return [ids[root] for root in roots]


def find_similar_pairs(
    fingerprints: Sequence[ImageFingerprint], max_distance: int
) -> list[SimilarPair]:
    """Pairs within ``max_distance`` pHash bits (full or cropped, direct or mirrored)."""
    full = np.array([fp.phash for fp in fingerprints], dtype=np.uint64)
    cropped = np.array([fp.phash_cropped for fp in fingerprints], dtype=np.uint64)
    pairs: list[SimilarPair] = []
    for index, fingerprint in enumerate(fingerprints[:-1]):
        later_full, later_cropped = full[index + 1 :], cropped[index + 1 :]
        distances = np.minimum.reduce(
            [
                hamming_distances(fingerprint.phash, later_full),
                hamming_distances(fingerprint.phash_flipped, later_full),
                hamming_distances(fingerprint.phash_cropped, later_cropped),
                hamming_distances(fingerprint.phash_cropped_flipped, later_cropped),
            ]
        )
        for offset in np.flatnonzero(distances <= max_distance):
            other = index + 1 + int(offset)
            correlation = thumbnail_correlation(fingerprint, fingerprints[other])
            pairs.append(SimilarPair(index, other, correlation))
    return pairs


def group_duplicates(
    fingerprints: Sequence[ImageFingerprint],
    pairs: Sequence[SimilarPair],
    duplicate_correlation: float,
    related_correlation: float,
) -> Grouping:
    """Cluster images into duplicates and related groups.

    Duplicates: identical decoded pixels, or correlation ``>= duplicate_correlation`` across
    any labels (so mislabeled copies surface). Related groups add same-label pairs with
    correlation ``>= related_correlation``; augmented images add no related edges, since
    they are removed and must not bridge two originals.
    """
    if related_correlation > duplicate_correlation:
        raise ValueError("related_correlation must be <= duplicate_correlation")

    duplicates = UnionFind(len(fingerprints))
    related = UnionFind(len(fingerprints))

    def link(first: int, second: int, duplicate: bool) -> None:
        related.union(first, second)
        if duplicate:
            duplicates.union(first, second)

    first_seen: dict[str, int] = {}
    for index, fingerprint in enumerate(fingerprints):
        if fingerprint.pixel_sha256 in first_seen:
            link(first_seen[fingerprint.pixel_sha256], index, duplicate=True)
        else:
            first_seen[fingerprint.pixel_sha256] = index

    for pair in pairs:
        first, second = fingerprints[pair.first], fingerprints[pair.second]
        if pair.correlation >= duplicate_correlation:
            link(pair.first, pair.second, duplicate=True)
        elif (
            pair.correlation >= related_correlation
            and first.label == second.label
            and not (first.is_augmented or second.is_augmented)
        ):
            link(pair.first, pair.second, duplicate=False)

    return Grouping(duplicates.component_ids(), related.component_ids())
