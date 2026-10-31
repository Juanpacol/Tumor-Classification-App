"""Calibrate the quality thresholds on the leakage-free Train split.

``python -m src.training.thresholds --data-dir src/data --manifest <manifest.csv>``

Each minimum sits in the gap between synthetic unusable references and the
lowest value over the Train images, so every Train image passes and every
reference fails.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.contracts import QualityMetrics
from src.errors import UnsupportedImageError
from src.preprocessing.loading import load_mri_image
from src.preprocessing.quality import QualityThresholds, measure_quality
from src.training.dataset import load_samples, split_samples

_SEED = 0
_NOISE_SIGMA = 0.2
_POSTERIZE_LEVELS = 4


@dataclass(frozen=True)
class Gap:
    """Lowest calibration value and highest reference value of one metric."""

    low: float
    high: float


@dataclass(frozen=True)
class Calibration:
    """Calibrated thresholds and where they come from."""

    thresholds: QualityThresholds
    images_used: int
    snr: Gap
    entropy: Gap


def measure(paths: Iterable[Path]) -> list[tuple[Path, QualityMetrics]]:
    """Quality of every readable image; unreadable images are skipped."""
    measured = []
    for path in paths:
        try:
            gray = load_mri_image(path)
        except UnsupportedImageError:
            continue
        measured.append((path, measure_quality(gray, skull_stripped=False)))
    return measured


def anchor_path(measured: list[tuple[Path, QualityMetrics]]) -> Path:
    """The image with the median SNR (ties by path), used to build degraded references."""
    ordered = sorted(measured, key=lambda item: (item[1].snr, str(item[0])))
    return ordered[len(ordered) // 2][0]


def reference_images(anchor: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Unusable images that must fail each metric, built with a fixed seed."""
    rng = np.random.default_rng(_SEED)
    blank = np.zeros_like(anchor)
    uniform = rng.random(anchor.shape).astype(np.float32)
    noisy = np.clip(anchor + rng.normal(0, _NOISE_SIGMA, anchor.shape), 0, 1)
    levels = _POSTERIZE_LEVELS
    posterized = np.minimum(np.floor(anchor * levels), levels - 1) / levels
    return {
        "snr": [blank, uniform, noisy.astype(np.float32)],
        "entropy": [blank, posterized.astype(np.float32)],
    }


def gap_thresholds(
    measured: list[QualityMetrics], references: dict[str, list[np.ndarray]]
) -> Calibration:
    """Thresholds in the gap of each metric: geometric midpoint for SNR, arithmetic for entropy.

    Raises ``ValueError`` when a metric has no gap.
    """
    gaps = {}
    for metric, images in references.items():
        low = min(getattr(m, metric) for m in measured)
        high = max(getattr(measure_quality(i, False), metric) for i in images)
        gaps[metric] = Gap(low, high)
    snr = _inside(gaps["snr"], math.sqrt(gaps["snr"].low * gaps["snr"].high), "snr")
    entropy = _inside(
        gaps["entropy"], (gaps["entropy"].low + gaps["entropy"].high) / 2, "entropy"
    )
    return Calibration(
        QualityThresholds(min_snr=snr, min_entropy=entropy),
        len(measured),
        gaps["snr"],
        gaps["entropy"],
    )


def _inside(gap: Gap, midpoint: float, metric: str) -> float:
    """``midpoint`` rounded to 2 decimals when that stays in (high, low]; raises without a gap."""
    if not gap.high < midpoint <= gap.low:
        raise ValueError(
            f"no gap for {metric}: lowest calibration value {gap.low:.4f} is not above "
            f"the highest reference value {gap.high:.4f}"
        )
    rounded = round(midpoint, 2)
    return rounded if gap.high < rounded <= gap.low else midpoint


def calibrate(paths: Iterable[Path]) -> Calibration:
    """Measure the calibration images, build the references and place the thresholds."""
    measured = measure(paths)
    if not measured:
        raise ValueError("no readable images to calibrate on")
    anchor = load_mri_image(anchor_path(measured))
    return gap_thresholds([m for _, m in measured], reference_images(anchor))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="cleaning manifest.csv"
    )
    args = parser.parse_args(argv)

    train = split_samples(load_samples(args.data_dir, args.manifest)).train
    if not train:
        parser.error("no Train images left after removing groups shared with Test")
    result = calibrate(s.path for s in train)
    print(
        json.dumps(
            {
                "min_snr": result.thresholds.min_snr,
                "min_entropy": result.thresholds.min_entropy,
                "images_used": result.images_used,
                "gap": {"snr": asdict(result.snr), "entropy": asdict(result.entropy)},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
