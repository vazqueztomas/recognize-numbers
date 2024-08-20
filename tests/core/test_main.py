import pytest
import cv2
import numpy as np
from recognize.core.main import preprocess_image, extract_numbers


def test_extract_numbers():
    # Create a dummy image with numbers for testing
    dummy_image = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(
        dummy_image,
        "123 456 789",
        (5, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    processed_image = preprocess_image(dummy_image)
    numbers = extract_numbers(processed_image)
    assert numbers == [
        123,
        456,
        789,
    ], "Extracted numbers should match the expected output"


if __name__ == "__main__":
    pytest.main()
