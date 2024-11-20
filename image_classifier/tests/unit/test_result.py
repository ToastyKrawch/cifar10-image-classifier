# tests/test_result.py
"""
Tests for the result handling in `result.py`.
Includes tests for model loading and prediction.
"""

from image_classifier.app import class_names
from image_classifier.result import load_and_predict


def test_model_load_and_predict():
    """Test if the model loads correctly and can make predictions."""

    prediction = load_and_predict(
        "image_classifier/model/image_classifier.keras",
        "image_classifier/tests/test_image.png",
    )
    assert prediction.shape == (1, len(class_names))
