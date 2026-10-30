import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.loading import load_mri_image
from src.preprocessing.quality import check_quality, measure_quality
from src.tests.synthetic import save, synthetic_slice
from src.training import thresholds
from src.training.thresholds import calibrate, main

HEADER = ["path", "split", "label", "group_id", "group_size", "keep", "reason"]


@pytest.fixture()
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Train 0-4 and Test 0-1; Train/1 shares a group with Test/0, Train/4 is dropped."""
    data_dir = tmp_path / "data"
    rows = []
    for index in range(5):
        relative = f"Train/glioma/{index}.png"
        save(synthetic_slice(index, size=128), data_dir / relative)
        keep = 0 if index == 4 else 1
        rows.append([relative, "Train", "glioma", index, 1, keep, "kept"])
    for index, group in enumerate((1, 10)):
        relative = f"Test/glioma/{index}.png"
        save(synthetic_slice(20 + index, size=128), data_dir / relative)
        rows.append([relative, "Test", "glioma", group, 1, 1, "kept"])
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return data_dir, manifest


def train_paths(data_dir: Path) -> list[Path]:
    return sorted((data_dir / "Train").rglob("*.png"))


def test_calibrated_thresholds_accept_the_calibration_images(
    dataset: tuple[Path, Path],
) -> None:
    paths = train_paths(dataset[0])

    found, used = calibrate(paths, percentile=0.0)

    assert used == len(paths)
    for path in paths:
        check_quality(measure_quality(load_mri_image(path), False), found)


def test_unreadable_images_are_skipped(dataset: tuple[Path, Path]) -> None:
    bad = dataset[0] / "Train" / "glioma" / "bad.png"
    bad.write_bytes(b"not an image")
    paths = train_paths(dataset[0])

    _, used = calibrate(paths)

    assert used == len(paths) - 1


def test_no_readable_images_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no readable images"):
        calibrate([tmp_path / "missing.png"])


def test_cli_uses_only_the_purged_train_split(
    dataset: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, manifest = dataset
    opened: list[Path] = []

    def recording_loader(path: Path) -> np.ndarray:
        opened.append(path)
        return load_mri_image(path)

    monkeypatch.setattr(thresholds, "load_mri_image", recording_loader)

    main(["--data-dir", str(data_dir), "--manifest", str(manifest)])

    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"min_snr", "min_entropy", "percentile", "images_used"}
    assert output["images_used"] == 3
    assert np.isfinite(output["min_snr"])
    assert sorted(p.name for p in opened) == ["0.png", "2.png", "3.png"]
    assert all(p.parent.parent.name == "Train" for p in opened)


def test_cli_explains_an_empty_purged_train_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    rows = []
    for split in ("Train", "Test"):
        relative = f"{split}/glioma/0.png"
        save(synthetic_slice(0, size=64), data_dir / relative)
        rows.append([relative, split, "glioma", 0, 1, 1, "kept"])
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)

    with pytest.raises(SystemExit):
        main(["--data-dir", str(data_dir), "--manifest", str(manifest)])

    assert "no Train images left" in capsys.readouterr().err


def test_cli_requires_a_manifest(dataset: tuple[Path, Path]) -> None:
    with pytest.raises(SystemExit):
        main(["--data-dir", str(dataset[0])])
