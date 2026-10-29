import csv
import json
from pathlib import Path

from src.preprocessing.deduplication import run


def test_run_writes_manifest_and_summary(dataset: Path) -> None:
    output_dir = run(dataset)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    with (output_dir / "manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paths = {row["path"] for row in rows}

    assert output_dir.parent == dataset / "processed"
    assert summary["total_images"] == 11
    assert summary["kept_images"] == 5
    assert summary["removed_by_reason"] == {
        "duplicate": 3,
        "augmented": 1,
        "label_conflict": 2,
    }
    assert summary["groups_spanning_splits_before"] == 3
    assert summary["groups_spanning_splits_after"] == 0
    assert len(rows) == 11
    assert all("\\" not in path for path in paths)
    duplicate_targets = [
        r["reason"].split(":", 1)[1]
        for r in rows
        if r["reason"].startswith("duplicate_of:")
    ]
    assert len(duplicate_targets) == 3 and set(duplicate_targets) <= paths
    assert (output_dir / "review_pairs.csv").exists()


def test_consecutive_runs_do_not_overwrite(dataset: Path) -> None:
    assert run(dataset) != run(dataset)
