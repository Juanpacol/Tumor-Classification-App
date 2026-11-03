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
CONFIG_FILE = "config.json"


def build_report(
    created_at: str,
    version: str,
    seed: int,
    dataset: dict[str, Any],
    validation: dict[str, Any],
    selection: dict[str, Any],
    ablation: dict[str, float],
    augmentation: dict[str, Any],
    cv_out_of_fold: dict[str, Any],
    test: dict[str, Any],
    calibration_method: str,
    shortcut_audit: dict[str, Any],
    guarantees: list[dict[str, Any]],
    config: dict[str, Any],
    config_source: dict[str, str] | None,
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
        "augmentation": augmentation,
        "cv_out_of_fold": cv_out_of_fold,
        "calibration_method": calibration_method,
        "test": test,
        "shortcut_audit": shortcut_audit,
        "leakage_guarantees": guarantees,
        "config": config,
        "config_source": config_source,
    }


def write_report(
    directory: Path, report: dict[str, Any], folds: Sequence[FoldRow]
) -> None:
    """Write ``report.json``, ``report.md``, the resolved ``config.json`` and the fold
    record ``folds.csv`` to a new folder (never overwrites)."""
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    (directory / CONFIG_FILE).write_text(json.dumps(report["config"], indent=2))
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
        *_per_class_lines(test),
        "## Confusion matrix (test, rows are the truth)",
        "",
        *_confusion_lines(test),
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
        *_out_of_fold_lines(report["cv_out_of_fold"]),
        *_augmentation_lines(report["augmentation"]),
        *_shortcut_lines(report["shortcut_audit"]),
        *_guarantee_lines(report["leakage_guarantees"]),
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
        *_config_lines(report["config_source"]),
    ]
    return "\n".join(lines)


def _per_class_lines(metrics: dict[str, Any]) -> list[str]:
    return [
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
        *(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |"
            for name, m in metrics["per_class"].items()
        ),
        "",
    ]


def _confusion_lines(metrics: dict[str, Any]) -> list[str]:
    return [
        "| | " + " | ".join(metrics["per_class"]) + " |",
        "|---|" + "---|" * len(metrics["per_class"]),
        *(
            f"| {name} | " + " | ".join(str(v) for v in row) + " |"
            for name, row in zip(metrics["per_class"], metrics["confusion_matrix"])
        ),
        "",
    ]


def _out_of_fold_lines(metrics: dict[str, Any]) -> list[str]:
    setting = "with" if metrics["augmented"] else "without"
    return [
        "## Per-class metrics (cross-validated on Train, out-of-fold, uncalibrated)",
        "",
        (
            f"Selected classifier, {setting} augmented copies. Every usable Train image "
            "is predicted once, by the model of the fold that validates it, and the "
            "counts are pooled over folds. Compare tuning rounds on this section, not on "
            "Test."
        ),
        "",
        *_per_class_lines(metrics),
        "Confusion matrix (rows are the truth):",
        "",
        *_confusion_lines(metrics),
    ]


def _config_lines(source: dict[str, str] | None) -> list[str]:
    origin = (
        f"round file `{source['path']}` (sha256 `{source['sha256'][:12]}`)"
        if source
        else "the defaults (no round file)"
    )
    return [
        "## Configuration",
        "",
        f"Parameters from {origin}. Every resolved value is in `{CONFIG_FILE}`.",
        "",
    ]


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


def _augmentation_lines(augmentation: dict[str, Any]) -> list[str]:
    verdict = "on" if augmentation["on"] else "off"
    copies = augmentation["copies"]
    without, with_copies = augmentation["cv_without"], augmentation["cv_with"]
    return [
        "## Augmentation (cross-validated macro-F1 on Train, selected classifier)",
        "",
        "| Fit rows | Macro-F1 mean | Macro-F1 std |",
        "|---|---|---|",
        f"| originals | {without['mean']:.3f} | {without['std']:.3f} |",
        (
            f"| originals + {copies} {'copy' if copies == 1 else 'copies'} per image "
            f"| {with_copies['mean']:.3f} | {with_copies['std']:.3f} |"
        ),
        "",
        (
            f"Augmentation **{verdict}**: it stays on only if it gains at least "
            f"{augmentation['margin']:.3f} (the std without copies). Copies vary the "
            "CNN block only. Ranges: "
            + ", ".join(f"{k} {v}" for k, v in augmentation["ranges"].items())
            + "."
        ),
        "",
    ]


def _shortcut_lines(audit: dict[str, Any]) -> list[str]:
    baseline = audit["metadata_baseline"]
    verdict = (
        "**WARNING: at or above the warning level. Present this result to the team "
        "before publishing a model.**"
        if baseline["warning"]
        else "Below the warning level."
    )
    return [
        "## Shortcut audit",
        "",
        (
            f"A {baseline['classifier']} that sees only "
            f"{', '.join(baseline['features'])} reaches a cross-validated balanced "
            f"accuracy of **{baseline['mean_balanced_accuracy']:.3f}** "
            f"(std {baseline['std_balanced_accuracy']:.3f}) on the same folds as the "
            f"model. Warning level {baseline['warning_level']:.2f}. {verdict}"
        ),
        "",
        "| no_tumor images in Test | Recall | Support |",
        "|---|---|---|",
        *(
            f"| {name} | {_recall(group['recall'])} | {group['support']} |"
            for name, group in audit["no_tumor_test_recall_by_source_size"].items()
        ),
        "",
    ]


def _guarantee_lines(guarantees: list[dict[str, Any]]) -> list[str]:
    return [
        "## Leakage guarantees",
        "",
        "| # | Guarantee | Status |",
        "|---|---|---|",
        *(f"| {g['id']} | {g['guarantee']} | {g['status']} |" for g in guarantees),
        "",
    ]


def _recall(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _inline(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key} {item}" for key, item in value.items())
    return str(value)
