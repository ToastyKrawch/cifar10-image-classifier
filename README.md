# CIFAR-10 Image Classifier — TensorFlow, Flask, and Software Testing

A Python project combining a convolutional neural network (CNN) trained on CIFAR-10 with a Flask web application for image classification. The repository also includes unit, integration, acceptance, and load-testing examples, a Dockerfile, and GitHub Actions CI.

**Status:** Academic project. Inference and training have not been re-verified as part of this documentation update.

## What it does

- Trains a CNN to classify images into ten CIFAR-10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
- Accepts an image through a browser form, resizes it to 32 × 32 pixels, normalizes the values, and displays the predicted class.
- Includes pytest-based tests and Locust load-test scenarios.
- Supports running the Flask application in a Docker container.

## Technologies

Python · TensorFlow/Keras · Flask · OpenCV · NumPy · Matplotlib · pytest · Locust · Docker · GitHub Actions

## Project layout

```text
image_classifier/
  app.py                Flask upload and inference endpoint
  images.py             CIFAR-10 loading and display utilities
  model_creation.py     CNN architecture and training entry point
  result.py             Standalone prediction helper
  model/                Saved Keras model
  templates/            Flask user interface
  tests/                Unit, integration, and acceptance tests
.github/workflows/ci.yml
Dockerfile
locustfile.py
requirements.txt
```

## Run locally

Python 3.10 is the version configured in the CI workflow. Create an isolated environment and install dependencies:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

Start the web app from the repository root:

```bash
python -m flask --app image_classifier.app run
```

Open http://127.0.0.1:5000. The saved model is tracked at `image_classifier/model/image_classifier.keras`.

To retrain and replace that model locally:

```bash
python -m image_classifier.model_creation
```

Training downloads CIFAR-10 data on first use and can take significant time. The training script uses 20,000 training examples and 4,000 test examples with ten epochs; results may vary by environment.

## Testing

```bash
python -m pytest
```

The test suite includes checks for Flask routes, image uploads, model shape/training, and image prediction. These tests may load TensorFlow and the saved model. Use Locust for HTTP load testing:

```bash
locust -f locustfile.py
```

Some acceptance tests assert a particular category for a fixed test image, so they depend on the specific committed model. The CI workflow no longer publishes a Docker image automatically.

## Docker

```bash
docker build -t cifar10-image-classifier .
docker run --rm -p 5000:5000 cifar10-image-classifier
```

## Notes

This is an academic demonstration, not a production image-classification service. The ten class labels are CIFAR-10 classes; predictions for unrelated images are not meaningful. Files uploaded to the browser interface are processed in memory rather than written to a shared tracked path.
