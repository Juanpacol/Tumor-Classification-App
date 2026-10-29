from pathlib import Path

import numpy as np
import pytest

from src.errors import QualityCheckError
from src.preprocessing.loading import load_mri_image
from src.preprocessing.prepare import PreprocessingConfig, prepare_image
from src.preprocessing.quality import QualityThresholds
from src.tests.synthetic import head_phantom, synthetic_slice

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def slice_array(seed: int = 1, size: int = 256) -> np.ndarray:
    return np.asarray(synthetic_slice(seed, size), dtype=np.float32) / 255.0


def test_outputs_have_the_expected_shape_and_type() -> None:
    result = prepare_image(slice_array())

    assert result.radiomics_image.shape == (256, 256)
    assert result.radiomics_image.dtype == np.float32
    assert result.cnn_input.shape == (3, 224, 224)
    assert result.cnn_input.dtype == np.float32


def test_cnn_input_is_normalized_per_channel() -> None:
    config = PreprocessingConfig(cnn_mean=(0.5, 0.5, 0.5), cnn_std=(0.25, 0.25, 0.25))
    gray = slice_array()

    result = prepare_image(gray, config)

    channels = result.cnn_input
    assert np.allclose(channels[0], channels[1]) and np.allclose(
        channels[1], channels[2]
    )
    assert channels.mean() == pytest.approx((gray.mean() - 0.5) / 0.25, abs=0.1)


def test_cnn_size_is_configurable_and_small_images_are_upscaled() -> None:
    result = prepare_image(slice_array(size=100), PreprocessingConfig(cnn_size=128))

    assert result.cnn_input.shape == (3, 128, 128)


def test_skull_stripping_is_off_by_default() -> None:
    gray = head_phantom()

    result = prepare_image(
        gray, PreprocessingConfig(thresholds=QualityThresholds(0, 0))
    )

    assert result.quality.skull_stripped is False
    assert np.array_equal(result.radiomics_image, gray)


def test_skull_stripping_when_enabled_masks_the_radiomics_image() -> None:
    gray = head_phantom()
    config = PreprocessingConfig(
        skull_stripping=True, thresholds=QualityThresholds(0, 0)
    )

    result = prepare_image(gray, config)

    assert result.quality.skull_stripped is True
    assert result.radiomics_image[128, int(128 + 0.4 * 256)] == 0.0


def test_stripping_fallback_is_reported_not_raised() -> None:
    flat_noise = 0.5 + 0.01 * np.random.default_rng(0).random((128, 128)).astype(
        np.float32
    )
    config = PreprocessingConfig(
        skull_stripping=True, thresholds=QualityThresholds(0, 0)
    )

    result = prepare_image(flat_noise, config)

    assert result.quality.skull_stripped is False


def test_low_quality_image_raises_with_metrics() -> None:
    with pytest.raises(QualityCheckError) as error:
        prepare_image(np.zeros((128, 128), dtype=np.float32))

    assert error.value.metrics.entropy == 0.0


def test_quality_is_measured_before_stripping() -> None:
    blank = np.zeros((128, 128), dtype=np.float32)

    with pytest.raises(QualityCheckError):
        prepare_image(blank, PreprocessingConfig(skull_stripping=True))


@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="dataset not available in src/data")
def test_real_dataset_sample_passes() -> None:
    sample = next(path for path in sorted(DATA_DIR.rglob("*.jpg")))

    prepare_image(load_mri_image(sample))
