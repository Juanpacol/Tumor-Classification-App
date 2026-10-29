import numpy as np

from src.preprocessing.cropping import crop_to_foreground


def test_crops_to_bounding_box_of_bright_pixels() -> None:
    image = np.zeros((20, 30), dtype=np.uint8)
    image[5:10, 8:20] = 200

    cropped = crop_to_foreground(image, threshold=10)

    assert cropped.shape == (5, 12)
    assert (cropped == 200).all()


def test_returns_input_when_image_is_empty() -> None:
    image = np.zeros((8, 8), dtype=np.uint8)

    assert crop_to_foreground(image) is image
