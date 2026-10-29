"""Summary statistics and output files of a cleaning run."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from src.preprocessing.deduplication.fingerprint import ImageFingerprint
from src.preprocessing.deduplication.matching import SimilarPair
from src.preprocessing.deduplication.selection import Decision


def _groups_spanning_splits(
    fingerprints: Sequence[ImageFingerprint], decisions: Sequence[Decision], kept_only: bool
) -> int:
    splits: dict[int, set[str]] = defaultdict(set)
    for fingerprint, decision in zip(fingerprints, decisions, strict=True):
        if decision.keep or not kept_only:
            splits[decision.group_id].add(fingerprint.split)
    return sum(1 for group_splits in splits.values() if len(group_splits) > 1)


def summarize(
    fingerprints: Sequence[ImageFingerprint],
    decisions: Sequence[Decision],
    params: dict[str, object],
) -> dict[str, object]:
    """Before/after counts, removal reasons and Train/Test leakage per related group."""
    kept = [fp for fp, d in zip(fingerprints, decisions, strict=True) if d.keep]
    return {
        "params": params,
        "total_images": len(fingerprints),
        "kept_images": len(kept),
        "removed_by_reason": dict(Counter(d.reason for d in decisions if not d.keep)),
        "largest_group_size_kept": max((d.group_size for d in decisions), default=0),
        "groups_spanning_splits_before": _groups_spanning_splits(fingerprints, decisions, False),
        "groups_spanning_splits_after": _groups_spanning_splits(fingerprints, decisions, True),
        "counts_before": dict(
            sorted(Counter(f"{fp.split}/{fp.label}" for fp in fingerprints).items())
        ),
        "counts_after": dict(sorted(Counter(f"{fp.split}/{fp.label}" for fp in kept).items())),
    }


def write_outputs(
    output_dir: Path,
    data_dir: Path,
    fingerprints: Sequence[ImageFingerprint],
    decisions: Sequence[Decision],
    review_pairs: Sequence[SimilarPair],
    summary: dict[str, object],
) -> None:
    """Write ``manifest.csv``, ``review_pairs.csv`` and ``summary.json`` to a new folder.

    Raises:
        FileExistsError: If ``output_dir`` exists, so earlier runs are never overwritten.
    """
    output_dir.mkdir(parents=True)

    def relative(index: int) -> str:
        return fingerprints[index].path.relative_to(data_dir).as_posix()

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "split", "label", "group_id", "group_size", "keep", "reason"])
        for index, (fp, decision) in enumerate(zip(fingerprints, decisions, strict=True)):
            reason = decision.reason
            if decision.keeper is not None:
                reason = f"duplicate_of:{relative(decision.keeper)}"
            writer.writerow(
                [
                    relative(index),
                    fp.split,
                    fp.label,
                    decision.group_id,
                    decision.group_size,
                    int(decision.keep),
                    reason,
                ]
            )

    with (output_dir / "review_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_a", "label_a", "path_b", "label_b", "correlation"])
        for pair in sorted(review_pairs, key=lambda p: -p.correlation):
            writer.writerow(
                [
                    relative(pair.first),
                    fingerprints[pair.first].label,
                    relative(pair.second),
                    fingerprints[pair.second].label,
                    f"{pair.correlation:.4f}",
                ]
            )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
