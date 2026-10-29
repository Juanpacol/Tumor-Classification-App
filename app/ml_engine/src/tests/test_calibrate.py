import json
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.calibrate import calibrate, main
from src.preprocessing.loading import load_mri_image
from src.preprocessing.quality import check_quality, measure_quality
from src.tests.synthetic import save, synthetic_slice


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    for split in ("Train", "Test"):
        for index in range(5):
            save(
                synthetic_slice(index, size=128),
                tmp_path / split / "glioma" / f"{index}.png",
            )
    return tmp_path


def test_calibrated_thresholds_accept_the_calibration_images(dataset: Path) -> None:
    paths = sorted(dataset.rglob("*.png"))

    thresholds, used = calibrate(paths, percentile=0.0)

    assert used == len(paths)
    for path in paths:
        check_quality(measure_quality(load_mri_image(path), False), thresholds)


def test_unreadable_images_are_skipped(dataset: Path) -> None:
    bad = dataset / "Train" / "glioma" / "bad.png"
    bad.write_bytes(b"not an image")
    paths = sorted(dataset.rglob("*.png"))

    _, used = calibrate(paths)

    assert used == len(paths) - 1


def test_no_readable_images_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no readable images"):
        calibrate([tmp_path / "missing.png"])


def test_cli_prints_the_thresholds_as_json(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--data-dir", str(dataset)])

    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"min_snr", "min_entropy", "percentile", "images_used"}
    assert output["images_used"] == 10
    assert np.isfinite(output["min_snr"])
