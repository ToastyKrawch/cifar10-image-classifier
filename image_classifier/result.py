"""
result.py: This script loads a pre-trained CNN model for CIFAR-10, preprocesses an image,
and makes a prediction about the image's class.
"""

# pylint: disable=import-error no-name-in-module

import cv2 as cv
import numpy as np
from tensorflow.keras import models  # type: ignore

from image_classifier.images import class_names



def load_and_predict(model_path, image_path):
    """
    Loads a pre-trained model and uses it to make a prediction on a given image.

    Args:
        modelPath (str): The file path to the pre-trained Keras model.
        imagepath (str): The file path to the input image for prediction.

    Returns:
        numpy.ndarray: The model's prediction for the input image.

    The function performs the following steps:
    - Loads a pre-trained model from the specified path.
    - Reads the image from the specified file path using OpenCV.
    - Resizes the image to 32x32 pixels.
    - Passes the preprocessed image through the model to get the prediction.
    """
    # Constructing neural net

    # Load the pre-trained model

    model = models.load_model(model_path)

    # Load and preprocess the image to make a prediction
    img = cv.imread(image_path)  # pylint: disable=no-member
    if img is None:
        raise ValueError("Cannot load image: Invalid or corrupted file")

    img = cv.resize(img, (32, 32))  # pylint: disable=no-member
    img = img / 255.00  # Normalize the pixel values to be between 0 and 1
    img = img[None, :]

    # Make a prediction
    return model.predict(img)


if __name__ == "__main__":

    prediction = load_and_predict(
        "image_classifier/model/image_classifier.keras",
        "image_classifier/tests/test_image.png",
    )

    index = np.argmax(prediction)
    print(f"Prediction is: {class_names[index]}")
