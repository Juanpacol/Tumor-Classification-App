"""End-to-end cleaning run: fingerprint, match, select, report."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from src.preprocessing.deduplication.discovery import discover_images
from src.preprocessing.deduplication.fingerprint import compute_fingerprint
from src.preprocessing.deduplication.matching import (
    find_similar_pairs,
    group_duplicates,
)
from src.preprocessing.deduplication.report import summarize, write_outputs
from src.preprocessing.deduplication.selection import select_representatives

REVIEW_CORRELATION = 0.90


def run(
    data_dir: Path,
    max_distance: int = 6,
    duplicate_correlation: float = 0.98,
    related_correlation: float = 0.95,
) -> Path:
    """Clean ``data_dir`` and return the new ``<data_dir>/processed/dedup_<timestamp>`` folder.

    Defaults were calibrated on the Nickparvar Kaggle dataset by visual review:
    correlation >= 0.98 was the same slice re-processed; 0.95-0.98 mixed re-processed
    copies and adjacent slices of one patient; cross-label pHash candidates at distance 6
    never reached 0.95. Pairs in neither group with correlation >= 0.90 go to review.

    Args:
        data_dir: Folder containing ``<Train|Test>/<label>/<image>``.
        max_distance: pHash Hamming distance for a pair to become a candidate.
        duplicate_correlation: Thumbnail correlation marking a duplicate (one kept).
        related_correlation: Thumbnail correlation grouping same-label slices (all kept).
    """
    paths, splits, labels = zip(*discover_images(data_dir))
    with ProcessPoolExecutor() as executor:
        fingerprints = list(
            executor.map(compute_fingerprint, paths, splits, labels, chunksize=64)
        )

    pairs = find_similar_pairs(fingerprints, max_distance)
    grouping = group_duplicates(
        fingerprints, pairs, duplicate_correlation, related_correlation
    )
    decisions = select_representatives(fingerprints, grouping)
    review_pairs = [
        pair
        for pair in pairs
        if grouping.group_ids[pair.first] != grouping.group_ids[pair.second]
        and pair.correlation >= REVIEW_CORRELATION
    ]
    summary = summarize(
        fingerprints,
        decisions,
        {
            "max_distance": max_distance,
            "duplicate_correlation": duplicate_correlation,
            "related_correlation": related_correlation,
        },
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = data_dir / "processed" / f"dedup_{timestamp}"
    write_outputs(output_dir, data_dir, fingerprints, decisions, review_pairs, summary)
    return output_dir
