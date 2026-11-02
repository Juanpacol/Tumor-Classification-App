"""Train the classifier: ``python -m src.training --data-dir src/data --manifest ...``."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

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
    return parser


def main(argv: list[str] | None = None) -> None:
    result = train(configure(build_parser().parse_args(argv)))
    test = result.report["test"]
    removed = result.report["dataset"]["split_removed_per_class"]
    print(
        "removed from Train (group in Test): "
        + ", ".join(f"{label} {count}" for label, count in removed.items())
    )
    print(f"artifact: {result.artifact_dir}")
    print(f"report:   {result.report_dir}")
    print(f"test macro-F1 {test['macro_f1']:.3f}, accuracy {test['accuracy']:.3f}")
    baseline = result.report["shortcut_audit"]["metadata_baseline"]
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
    if not result.report["target_met"]:
        print(
            f"Target macro-F1 {result.report['target_macro_f1']:.2f} not met: "
            "present the results to the team before continuing.",
            file=sys.stderr,
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
    )
    if args.config is not None:
        config = apply_round(config, args.config)
    flags = {"seed": args.seed, "n_splits": args.n_splits}
    return replace(config, **{k: v for k, v in flags.items() if v is not None})


if __name__ == "__main__":
    main()
