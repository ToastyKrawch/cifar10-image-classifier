# tests/test_result.py
"""
Tests for the result handling in `result.py`.
Includes tests for model loading and prediction.
"""

from app import class_names
from result import load_and_predict


def test_model_load_and_predict():
    """Test if the model loads correctly and can make predictions."""
    prediction = load_and_predict("model/image_classifier.keras", "test_image.png")
    assert prediction.shape == (1, len(class_names))
