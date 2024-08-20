import pytest


@pytest.fixture
def image_path() -> str:
    return "recognize/assets/tabla.jpeg"


@pytest.fixture
def not_existing_image_path() -> str:
    return "path/to/nonexistent/image.png"
