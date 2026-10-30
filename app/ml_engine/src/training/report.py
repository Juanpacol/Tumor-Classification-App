"""Timestamped training report in JSON and markdown."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from src.training.folds import FoldRow

TARGET_MACRO_F1 = 0.90
FOLDS_FILE = "folds.csv"


def build_report(
    created_at: str,
    version: str,
    seed: int,
    dataset: dict[str, Any],
    validation: dict[str, Any],
    selection: dict[str, Any],
    ablation: dict[str, float],
    test: dict[str, Any],
    calibration_method: str,
) -> dict[str, Any]:
    """Collect the results of one training run and whether the target was met."""
    return {
        "created_at": created_at,
        "version": version,
        "seed": seed,
        "target_macro_f1": TARGET_MACRO_F1,
        "target_met": test["macro_f1"] >= TARGET_MACRO_F1,
        "dataset": dataset,
        "validation": validation,
        "selection": selection,
        "ablation_macro_f1": ablation,
        "calibration_method": calibration_method,
        "test": test,
    }


def write_report(
    directory: Path, report: dict[str, Any], folds: Sequence[FoldRow]
) -> None:
    """Write ``report.json``, ``report.md`` and the fold record ``folds.csv`` to a new
    folder (never overwrites)."""
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    (directory / "report.md").write_text(render_markdown(report))
    with (directory / FOLDS_FILE).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, [f.name for f in fields(FoldRow)])
        writer.writeheader()
        writer.writerows({**asdict(row), "used": int(row.used)} for row in folds)


def render_markdown(report: dict[str, Any]) -> str:
    test = report["test"]
    verdict = "met" if report["target_met"] else "NOT met"
    lines = [
        f"# Training report {report['version']}",
        "",
        f"Created {report['created_at']}, seed {report['seed']}.",
        "",
        (
            f"**Test macro-F1 {test['macro_f1']:.3f}**, accuracy {test['accuracy']:.3f}, "
            f"balanced accuracy {test['balanced_accuracy']:.3f}. "
            f"Target {report['target_macro_f1']:.2f}: {verdict}."
        ),
        "",
        "## Per class (test)",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
        *(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |"
            for name, m in test["per_class"].items()
        ),
        "",
        "## Confusion matrix (test, rows are the truth)",
        "",
        "| | " + " | ".join(test["per_class"]) + " |",
        "|---|" + "---|" * len(test["per_class"]),
        *(
            f"| {name} | " + " | ".join(str(v) for v in row) + " |"
            for name, row in zip(test["per_class"], test["confusion_matrix"])
        ),
        "",
        *_validation_lines(report["validation"]),
        "## Classifier selection (cross-validated on Train, chosen by macro-F1)",
        "",
        "| Classifier | Macro-F1 mean | Macro-F1 std | Bal. acc. mean | Bal. acc. std |",
        "|---|---|---|---|---|",
        *(
            f"| {name} | {s['mean_macro_f1']:.3f} | {s['std_macro_f1']:.3f} "
            f"| {s['mean_balanced_accuracy']:.3f} | {s['std_balanced_accuracy']:.3f} |"
            for name, s in report["selection"]["scores"].items()
        ),
        "",
        (
            f"Selected: **{report['selection']['best']}**, calibrated with "
            f"{report['calibration_method']}."
        ),
        "",
        "## Ablation (cross-validated macro-F1 on Train)",
        "",
        "| Features | Macro-F1 |",
        "|---|---|",
        *(
            f"| {name} | {value:.3f} |"
            for name, value in report["ablation_macro_f1"].items()
        ),
        "",
        "## Data",
        "",
        *(
            f"- {key}: {_inline(value)}"
            for key, value in report["dataset"].items()
            if key != "skipped_reasons"
        ),
        "",
        "The reason for every skipped image is in `report.json`.",
        "",
    ]
    return "\n".join(lines)


def _validation_lines(validation: dict[str, Any]) -> list[str]:
    counts = validation["used_per_fold_and_class"]
    labels = list(next(iter(counts.values()), {}))
    return [
        "## Folds (used Train images per fold and class)",
        "",
        (
            f"{validation['method']}, {validation['n_splits']} folds, group "
            f"`{validation['group']}`, seed {validation['seed']}. "
            f"The fold of every Train image left after the leakage-free split is in `{FOLDS_FILE}`."
        ),
        "",
        "| Fold | " + " | ".join(labels) + " | Total |",
        "|---|" + "---|" * (len(labels) + 1),
        *(
            f"| {fold} | "
            + " | ".join(str(per_class[label]) for label in labels)
            + f" | {sum(per_class.values())} |"
            for fold, per_class in counts.items()
        ),
        "",
    ]


def _inline(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key} {item}" for key, item in value.items())
    return str(value)
