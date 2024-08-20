from recognize.core.main import load_image, preprocess_image
import numpy as np


def test_load_image(image_path) -> None:
    image = load_image(image_path)
    assert image is not None, "Image should be loaded successfully"


def test_load_image_with_invalid_image_path(not_existing_image_path) -> None:
    image = load_image(image_path=not_existing_image_path)
    assert image is None, "Image should not be loaded for an invalid path"


def test_preprocess_image():
    # Create a dummy image for testing
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    processed_image = preprocess_image(dummy_image)
    assert (
        processed_image.shape == dummy_image.shape[:2]
    ), "Processed image should have the same dimensions as the input image"
