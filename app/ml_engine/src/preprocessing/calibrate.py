"""Calibrate quality thresholds on a dataset: ``python -m src.preprocessing.calibrate``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from src.errors import UnsupportedImageError
from src.preprocessing.deduplication.discovery import discover_images
from src.preprocessing.loading import load_mri_image
from src.preprocessing.quality import QualityThresholds, estimate_snr, shannon_entropy


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
            gray = load_mri_image(path)
        except UnsupportedImageError:
            continue
        snrs.append(estimate_snr(gray))
        entropies.append(shannon_entropy(gray))
    if not snrs:
        raise ValueError("no readable images to calibrate on")
    thresholds = QualityThresholds(
        min_snr=float(np.percentile(snrs, percentile)),
        min_entropy=float(np.percentile(entropies, percentile)),
    )
    return thresholds, len(snrs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="folder with Train/ and Test/"
    )
    parser.add_argument("--percentile", type=float, default=1.0)
    args = parser.parse_args(argv)

    paths = [path for path, _, _ in discover_images(args.data_dir)]
    thresholds, used = calibrate(paths, args.percentile)
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
