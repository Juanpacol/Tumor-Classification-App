"""Decide which image of each duplicate cluster is kept."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from src.preprocessing.deduplication.fingerprint import ImageFingerprint
from src.preprocessing.deduplication.matching import Grouping


@dataclass(frozen=True)
class Decision:
    """Cleaning outcome for one image.

    ``reason`` is ``kept``, ``duplicate``, ``augmented`` or ``label_conflict``;
    ``keeper`` is the index of the kept image for duplicates. ``group_size`` counts the
    kept images of the image's related group.
    """

    group_id: int
    group_size: int
    keep: bool
    reason: str
    keeper: int | None = None


def _majority_label(fingerprints: Sequence[ImageFingerprint], indices: list[int]) -> str | None:
    counts = Counter(fingerprints[i].label for i in indices).most_common(2)
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def select_representatives(
    fingerprints: Sequence[ImageFingerprint], grouping: Grouping
) -> list[Decision]:
    """Keep one image per duplicate cluster.

    * Members whose label differs from the cluster's majority label are dropped
      (``label_conflict``); on a tie, the whole cluster is.
    * Augmentations are dropped (``augmented``): augmentation belongs after the split.
    * Among the rest, the largest image is kept and the others become ``duplicate``.
    """
    clusters: dict[int, list[int]] = defaultdict(list)
    for index, duplicate_id in enumerate(grouping.duplicate_ids):
        clusters[duplicate_id].append(index)

    outcomes: dict[int, tuple[bool, str, int | None]] = {}
    for indices in clusters.values():
        majority = _majority_label(fingerprints, indices)
        originals = [
            i
            for i in indices
            if fingerprints[i].label == majority and not fingerprints[i].is_augmented
        ]
        keeper = (
            min(originals, key=lambda i: (-fingerprints[i].area, fingerprints[i].path.as_posix()))
            if originals
            else None
        )
        for i in indices:
            if fingerprints[i].label != majority:
                outcomes[i] = (False, "label_conflict", None)
            elif fingerprints[i].is_augmented:
                outcomes[i] = (False, "augmented", None)
            elif i == keeper:
                outcomes[i] = (True, "kept", None)
            else:
                outcomes[i] = (False, "duplicate", keeper)

    kept_per_group = Counter(grouping.group_ids[i] for i, outcome in outcomes.items() if outcome[0])
    return [
        Decision(grouping.group_ids[i], kept_per_group[grouping.group_ids[i]], *outcomes[i])
        for i in range(len(fingerprints))
    ]
