"""Train the classifier: ``python -m src.training --data-dir src/data --manifest ...``.

Without ``--final`` the run is a tuning round: it never reads Testing and writes no
artifact. ``--final`` scores Testing once and writes the artifact of the kept round.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.training.config import TrainingConfig, apply_round
from src.training.run import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="cleaning manifest.csv"
    )
    parser.add_argument("--backbone-bundle", type=Path, default=Path("weights"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--config", type=Path, default=None, help="round file (TOML) with overrides"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="overrides the round file"
    )
    parser.add_argument(
        "--n-splits", type=int, default=None, help="overrides the round file"
    )
    parser.add_argument("--version", default="0.0.0-dev")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--final",
        action="store_true",
        help="score Testing once and write the artifact (only for the kept round)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    result = train(configure(build_parser().parse_args(argv)))
    report = result.report
    removed = report["dataset"]["split_removed_per_class"]
    print(
        "removed from Train (group in Test): "
        + ", ".join(f"{label} {count}" for label, count in removed.items())
    )
    print(f"report:   {result.report_dir}")
    if result.artifact_dir is None:
        _print_tuning(report)
    else:
        test = report["test"]
        print(f"artifact: {result.artifact_dir}")
        print(f"test macro-F1 {test['macro_f1']:.3f}, accuracy {test['accuracy']:.3f}")
    baseline = report["shortcut_audit"]["metadata_baseline"]
    print(
        f"metadata-only baseline balanced accuracy {baseline['mean_balanced_accuracy']:.3f}"
    )
    if baseline["warning"]:
        print(
            f"WARNING: the metadata-only baseline reaches {baseline['warning_level']:.2f} "
            "balanced accuracy or more, so the class can be told from the file alone. "
            "Present the result to the team before publishing a model.",
            file=sys.stderr,
        )
    if report["target_met"] is False:
        print(
            f"Target macro-F1 {report['target_macro_f1']:.2f} not met: "
            "present the results to the team before continuing.",
            file=sys.stderr,
        )


def _print_tuning(report: dict[str, Any]) -> None:
    best = report["selection"]["best"]
    score = report["selection"]["scores"][best]
    print(
        f"tuning run (Testing not read): {best}, CV macro-F1 "
        f"{score['mean_macro_f1']:.3f} +/- {score['std_macro_f1']:.3f}"
    )
    recall = {
        label: metrics["recall"]
        for label, metrics in report["cv_out_of_fold"]["per_class"].items()
    }
    print(
        "out-of-fold recall: "
        + ", ".join(f"{label} {value:.3f}" for label, value in recall.items())
    )


def configure(args: argparse.Namespace) -> TrainingConfig:
    """The run configuration: defaults, then the round file, then ``--seed`` and
    ``--n-splits`` when given."""
    config = TrainingConfig(
        data_dir=args.data_dir,
        manifest=args.manifest,
        backbone_bundle=args.backbone_bundle,
        out_dir=args.out,
        version=args.version,
        cache_dir=args.cache_dir,
        final=args.final,
    )
    if args.config is not None:
        config = apply_round(config, args.config)
    flags = {"seed": args.seed, "n_splits": args.n_splits}
    return replace(config, **{k: v for k, v in flags.items() if v is not None})


if __name__ == "__main__":
    main()
