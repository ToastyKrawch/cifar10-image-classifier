"""Flask application for classifying uploaded images using a CIFAR-10 CNN."""

import base64
from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

from image_classifier.images import class_names

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "model" / "image_classifier.keras"
model = tf.keras.models.load_model(str(MODEL_PATH))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def preprocess_image(img_bytes):
    """Decode and normalize an uploaded image without saving it to disk."""
    if not img_bytes:
        raise ValueError("Cannot load image. The file may be invalid or unsupported.")
    encoded = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv.imdecode(encoded, cv.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot load image. The file may be invalid or unsupported.")
    image = cv.resize(image, (32, 32))
    return np.expand_dims(image.astype(np.float32) / 255.0, axis=0)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or not file.filename:
            return "Please choose an image to upload.", 400
        try:
            raw = file.read()
            image = preprocess_image(raw)
        except ValueError as exc:
            return str(exc), 400
        prediction = model.predict(image)
        result = class_names[int(np.argmax(prediction))]
        # Embed the validated image directly; no shared, persistent upload path.
        image_base64 = base64.b64encode(raw).decode("ascii")
        return render_template("index.html", result=result, image_base64=image_base64)
    return render_template("index.html", result=None, image_base64=None)


if __name__ == "__main__":
    app.run()
