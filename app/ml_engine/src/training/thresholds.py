"""Calibrate the quality thresholds on the leakage-free Train split.

``python -m src.training.thresholds --data-dir src/data --manifest <manifest.csv>``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from src.errors import UnsupportedImageError
from src.preprocessing.loading import load_mri_image
from src.preprocessing.quality import QualityThresholds, measure_quality
from src.training.dataset import load_samples, split_samples


def calibrate(
    paths: Iterable[Path], percentile: float = 1.0
) -> tuple[QualityThresholds, int]:
    """Thresholds at the given percentile of each metric, and how many images were used.

    With the default 1st percentile, at least 99% of the calibration images pass.
    Unreadable images are skipped.
    """
    snrs: list[float] = []
    entropies: list[float] = []
    for path in paths:
        try:
            metrics = measure_quality(load_mri_image(path), skull_stripped=False)
        except UnsupportedImageError:
            continue
        snrs.append(metrics.snr)
        entropies.append(metrics.entropy)
    if not snrs:
        raise ValueError("no readable images to calibrate on")
    thresholds = QualityThresholds(
        min_snr=float(np.percentile(snrs, percentile)),
        min_entropy=float(np.percentile(entropies, percentile)),
    )
    return thresholds, len(snrs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="cleaning manifest.csv"
    )
    parser.add_argument("--percentile", type=float, default=1.0)
    args = parser.parse_args(argv)

    train = split_samples(load_samples(args.data_dir, args.manifest)).train
    if not train:
        parser.error("no Train images left after removing groups shared with Test")
    thresholds, used = calibrate((s.path for s in train), args.percentile)
    print(
        json.dumps(
            {
                "min_snr": thresholds.min_snr,
                "min_entropy": thresholds.min_entropy,
                "percentile": args.percentile,
                "images_used": used,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
