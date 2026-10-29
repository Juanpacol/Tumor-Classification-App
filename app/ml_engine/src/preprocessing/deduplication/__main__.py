"""Command line entry point: ``python -m src.preprocessing.deduplication``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.preprocessing.deduplication.pipeline import run


def main() -> None:
    """Parse arguments, run the cleaning and print the summary."""
    parser = argparse.ArgumentParser(
        description="Detect and remove duplicate MRI slices."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("src/data"))
    parser.add_argument("--max-distance", type=int, default=6)
    parser.add_argument("--duplicate-correlation", type=float, default=0.98)
    parser.add_argument("--related-correlation", type=float, default=0.95)
    args = parser.parse_args()

    output_dir = run(
        args.data_dir,
        args.max_distance,
        args.duplicate_correlation,
        args.related_correlation,
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"Output: {output_dir}")
    print(f"Kept {summary['kept_images']} of {summary['total_images']} images")
    print(f"Removed: {summary['removed_by_reason']}")
    print(
        "Groups spanning Train/Test: "
        f"{summary['groups_spanning_splits_before']} before, "
        f"{summary['groups_spanning_splits_after']} after"
    )


if __name__ == "__main__":
    main()
