# tests/test_model_creation.py
# pylint: disable=import-error
"""
Tests for the model creation and training process in `model_creation.py`.
Includes tests for model structure and training on a small dataset.
"""
import numpy as np  # type: ignore

import model_creation


def test_model_creation():
    """Test if the model can be created and compiled successfully."""
    model = model_creation.create_model()
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    assert model is not None


def test_model_training():
    """Test if the model can train on a small dummy dataset."""
    model = model_creation.create_model()
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Dummy data
    x_train = np.random.rand(10, 32, 32, 3)
    y_train = np.random.randint(0, 10, (10, 1))

    history = model.fit(x_train, y_train, epochs=1, verbose=0)
    assert history is not None
