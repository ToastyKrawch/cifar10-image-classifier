"""test_acceptance.py: Holds two acceptance tests for the application"""

from io import BytesIO

import pytest  # pylint: disable=import-error

from image_classifier.app import app


@pytest.fixture(name="client")
def client_fixture():
    """Fixture to create a test client for the Flask app."""
    with app.test_client() as client:
        yield client


def test_valid_image_classification(client):
    """
    Test that the system correctly classifies a valid image.
    """
    # Simulate uploading a valid image
    with open("image_classifier/tests/test_image.png", "rb") as img:
        data = {"file": (img, "test_image.png")}
        response = client.post("/", content_type="multipart/form-data", data=data)

    # Assert that the response contains the expected result
    assert response.status_code == 200
    assert b"car" in response.data


def test_invalid_image_upload(client):
    """
    Test that the system handles invalid file uploads gracefully.
    """
    # Simulate uploading an invalid file
    data = {"file": (BytesIO(b"invalid content"), "invalid_image.txt")}
    response = client.post("/", content_type="multipart/form-data", data=data)

    # Assert that the response returns an error or handles gracefully
    assert response.status_code == 400
    assert b"Cannot load image" in response.data  # Match the error message
