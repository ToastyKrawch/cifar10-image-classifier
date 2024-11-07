# tests/test_app.py
# pylint: disable=import-error no-name-in-module
"""
Tests for the Flask application in `app.py`.
Includes tests for app initialization, route handling, and image upload.
"""
import pytest

from app import app


@pytest.fixture(name="client")
def client_fixture():
    """Fixture to create a test client for the Flask app."""
    with app.test_client() as client:
        yield client


def test_app_initialization():
    """Test if the app initializes correctly."""
    assert app is not None


def test_index_route(client):
    """Test if the index route returns a 200 status code."""
    response = client.get("/")
    assert response.status_code == 200


def test_image_upload(client):
    """Test if an image upload is handled correctly."""
    with open("tests/test_image.png", "rb") as img:
        data = {"file": (img, "test_image.png")}
        response = client.post("/", content_type="multipart/form-data", data=data)
        assert response.status_code == 200
        assert (
            b"Result" in response.data
        )  # Check for the presence of "Result" in the response
