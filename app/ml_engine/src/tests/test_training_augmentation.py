import numpy as np

from src.training.augmentation import (
    AugmentationConfig,
    AugmentationParams,
    augment,
    copy_rng,
    draw,
)

CONFIG = AugmentationConfig()
NEUTRAL = AugmentationParams(0.0, False, 1.0, 1.0, 1.0)
DIGEST = "ab" * 32


def cnn_image(seed: int = 0) -> np.ndarray:
    image = np.random.default_rng(seed).random((224, 224), dtype=np.float32)
    image[:20] = 0.0
    return image


def test_neutral_parameters_return_the_source() -> None:
    image = cnn_image()

    assert np.array_equal(augment(image, NEUTRAL), image)


def test_a_copy_keeps_the_shape_and_stays_in_range() -> None:
    image = cnn_image()
    extreme = AugmentationParams(10.0, True, 1.1, 1.1, 1.1)

    copy = augment(image, extreme)

    assert copy.shape == (224, 224) and copy.dtype == np.float32
    assert copy.min() >= 0.0 and copy.max() <= 1.0
    assert not np.array_equal(copy, image)


def test_pixels_beyond_the_border_are_black() -> None:
    image = np.ones((224, 224), dtype=np.float32)

    copy = augment(image, AugmentationParams(0.0, False, 0.9, 1.0, 1.0))

    assert copy[0, 0] == 0.0 and copy[112, 112] > 0.0


def test_black_pixels_stay_black_at_low_contrast() -> None:
    image = np.full((224, 224), 0.5, dtype=np.float32)
    image[:, :40] = 0.0

    copy = augment(image, AugmentationParams(5.0, False, 0.9, 1.1, 0.9))

    assert copy[0, 0] == 0.0 and copy[112, 10] == 0.0
    assert copy[112, 150] > 0.0


def test_a_flip_alone_mirrors_the_image() -> None:
    image = cnn_image()

    copy = augment(image, AugmentationParams(0.0, True, 1.0, 1.0, 1.0))

    assert np.array_equal(copy, np.fliplr(image))


def test_drawn_parameters_stay_in_their_ranges() -> None:
    drawn = [draw(CONFIG, copy_rng(seed, DIGEST, 0)) for seed in range(500)]

    assert all(-10.0 <= p.rotation_deg <= 10.0 for p in drawn)
    assert all(0.9 <= p.scale <= 1.1 for p in drawn)
    assert all(0.9 <= p.brightness <= 1.1 for p in drawn)
    assert all(0.9 <= p.contrast <= 1.1 for p in drawn)
    flips = sum(p.flip for p in drawn)
    assert 200 < flips < 300


def test_same_seed_content_and_index_give_the_same_copy() -> None:
    image = cnn_image()

    first = augment(image, draw(CONFIG, copy_rng(42, DIGEST, 0)))
    second = augment(image, draw(CONFIG, copy_rng(42, DIGEST, 0)))

    assert np.array_equal(first, second)


def test_another_index_seed_or_content_gives_another_draw() -> None:
    base = draw(CONFIG, copy_rng(42, DIGEST, 0))

    assert draw(CONFIG, copy_rng(42, DIGEST, 1)) != base
    assert draw(CONFIG, copy_rng(43, DIGEST, 0)) != base
    assert draw(CONFIG, copy_rng(42, "cd" * 32, 0)) != base
