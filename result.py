# pylint: disable=import-error, no-name-in-module

"""
result.py: This script loads a pre-trained CNN model for CIFAR-10, preprocesses an image,
and makes a prediction about the image's class.
"""

import cv2 as cv
import numpy as np
from tensorflow.keras import models

from common import class_names, display_sample_images, load_cifar10_data

# Load dataset
(training_images, training_labels), (testing_images, testing_labels) = (
    load_cifar10_data()
)

# Display some sample images from the dataset
display_sample_images(training_images, training_labels, class_names)

# Constructing neural net


# Load the pre-trained model

model = models.load_model("model/image_classifier.keras")

# Load and preprocess the image to make a prediction

img = cv.imread("static/uploads/uploaded_image.png")  # pylint: disable=no-member

img = cv.resize(img, (32, 32))  # pylint: disable=no-member
img = img / 255.00  # Normalize the pixel values to be between 0 and 1
img = img[None, :]

# Make a prediction
prediction = model.predict(img)
index = np.argmax(prediction)
print(f"Prediction is: {class_names[index]}")
