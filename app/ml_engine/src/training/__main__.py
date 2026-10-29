"""Train the classifier: ``python -m src.training --data-dir src/data --manifest ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.training.model import DEFAULT_SEED, N_SPLITS
from src.training.run import TrainingConfig, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="cleaning manifest.csv"
    )
    parser.add_argument("--backbone-bundle", type=Path, default=Path("weights"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--version", default="0.0.0-dev")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    result = train(
        TrainingConfig(
            data_dir=args.data_dir,
            manifest=args.manifest,
            backbone_bundle=args.backbone_bundle,
            out_dir=args.out,
            seed=args.seed,
            n_splits=args.n_splits,
            version=args.version,
            cache_dir=args.cache_dir,
        )
    )
    test = result.report["test"]
    print(f"artifact: {result.artifact_dir}")
    print(f"report:   {result.report_dir}")
    print(f"test macro-F1 {test['macro_f1']:.3f}, accuracy {test['accuracy']:.3f}")
    if not result.report["target_met"]:
        print(
            f"Target macro-F1 {result.report['target_macro_f1']:.2f} not met: "
            "present the results to the team before continuing.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
