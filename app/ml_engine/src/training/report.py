"""Timestamped training report in JSON and markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TARGET_MACRO_F1 = 0.90


def build_report(
    created_at: str,
    version: str,
    seed: int,
    dataset: dict[str, Any],
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
        "selection": selection,
        "ablation_macro_f1": ablation,
        "calibration_method": calibration_method,
        "test": test,
    }


def write_report(directory: Path, report: dict[str, Any]) -> None:
    """Write ``report.json`` and ``report.md`` to a new folder (never overwrites)."""
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    (directory / "report.md").write_text(render_markdown(report))


def render_markdown(report: dict[str, Any]) -> str:
    test = report["test"]
    verdict = "met" if report["target_met"] else "NOT met"
    lines = [
        f"# Training report {report['version']}",
        "",
        f"Created {report['created_at']}, seed {report['seed']}.",
        "",
        (
            f"**Test macro-F1 {test['macro_f1']:.3f}**, accuracy {test['accuracy']:.3f}. "
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
        "## Classifier selection (cross-validated macro-F1 on Train)",
        "",
        "| Classifier | Mean | Std |",
        "|---|---|---|",
        *(
            f"| {name} | {s['mean_macro_f1']:.3f} | {s['std_macro_f1']:.3f} |"
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


def _inline(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key} {item}" for key, item in value.items())
    return str(value)
