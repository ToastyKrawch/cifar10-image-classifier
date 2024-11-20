# tests/test_integration.py pylint: disable=import-error
"""
This file contains the happy and sad path tests we would like to include on commit.
"""
import numpy as np
import pytest

from image_classifier.result import load_and_predict


def test_successful_prediction():
    """Test that a valid image input results in a successful prediction."""
    model_path = "image_classifier/model/image_classifier.keras"
    image_path = "image_classifier/tests/test_image.png"
    prediction = load_and_predict(model_path, image_path)
    assert prediction is not None, "Prediction should not be None"
    assert isinstance(
        prediction, (list, np.ndarray)
    ), "Prediction should be a list or numpy array"
    assert len(prediction) > 0, "Prediction should have at least one class probability"


def test_invalid_image_input():
    """Test that an invalid image input raises an appropriate error."""
    model_path = "image_classifier/model/image_classifier.keras"
    invalid_image_path = "tests/invalid_image.txt"
    with pytest.raises(ValueError) as excinfo:
        load_and_predict(model_path, invalid_image_path)

    assert "Cannot load image: Invalid or corrupted file" in str(excinfo.value)
