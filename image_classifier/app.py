"""
app.py: Flask web application for image classification using a pre-trained model
on the CIFAR-10 dataset. The application allows users to upload an image and
returns the predicted class.
"""

# pylint: disable=import-error no-name-in-module no-member

import cv2 as cv
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("image_classifier/model/image_classifier.keras")

# Class names for CIFAR-10
class_names = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def preprocess_image(img_path):
    """
    Preprocess the input image by resizing it to 32x32, normalizing the pixel values,
    and adding a batch dimension to make it compatible with the model.

    Args:
        img_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image ready for prediction.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    img = cv.imread(img_path)  # pylint: disable=no-member
    if img is None:  # Check if the image was successfully loaded
        raise ValueError("Cannot load image. The file may be invalid or unsupported.")
    img = cv.resize(img, (32, 32))  # pylint: disable=no-member
    img = img / 255.0
    img = img[None, :]
    return img


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handle the main page of the web application. On POST request, the function processes
    the uploaded image and returns the predicted class.

    Returns:
        str: Rendered HTML page with the classification result or an error message.
    """
    if request.method == "POST":
        # Handle the uploaded image
        file = request.files["file"]
        if file:
            try:
                # Save the uploaded image
                img_path = "image_classifier/static/uploads/uploaded_image.png"
                file.save(img_path)

                # Preprocess the image
                img = preprocess_image(img_path)

                # Make a prediction
                prediction = model.predict(img)
                predicted_class_index = np.argmax(prediction)
                result = class_names[predicted_class_index]

                return render_template("index.html", result=result, image_path=img_path)
            except ValueError as e:
                # Handle invalid image errors
                return str(e), 400  # Return a 400 Bad Request error
    return render_template("index.html", result=None, image_path=None)


if __name__ == "__main__":
    app.run()
