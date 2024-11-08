# pylint: disable=import-error
"""
model_creation.py: This script builds, trains, and evaluates a Convolutional Neural Network (CNN)
on the CIFAR-10 dataset, and saves the trained model as `image_classifier.keras`.
"""

# Import necessary libraries
from tensorflow.keras import (  # type: ignore #pylint: disable=import-error no-name-in-module
    layers,
    models,
)

from image_classifier.images import display_images, load_data


def create_model():
    """
    Creates and returns a Convolutional Neural Network (CNN) model using the Keras Sequential API.

    The model architecture is as follows:
    - Conv2D layer with 32 filters, 3x3 kernel size, ReLU activation, and input shape of (32, 32, 3)
    - MaxPooling2D layer with 2x2 pool size
    - Conv2D layer with 64 filters, 3x3 kernel size, and ReLU activation
    - MaxPooling2D layer with 2x2 pool size
    - Conv2D layer with 64 filters, 3x3 kernel size, and ReLU activation
    - Flatten layer to convert the 2D feature maps into a 1D feature vector
    - Dense layer with 64 units and ReLU activation
    - Output Dense layer with 10 units and softmax activation for classification

    Returns:
        tensorflow.keras.models.Sequential: A compiled CNN model ready for training.
    """
    model_creation = models.Sequential()
    model_creation.add(
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3))
    )
    model_creation.add(layers.MaxPooling2D((2, 2)))
    model_creation.add(
        layers.Conv2D(64, (3, 3), activation="relu")
    )  # No input_shape here
    model_creation.add(layers.MaxPooling2D((2, 2)))
    model_creation.add(
        layers.Conv2D(64, (3, 3), activation="relu")
    )  # No input_shape here
    model_creation.add(layers.Flatten())
    model_creation.add(layers.Dense(64, activation="relu"))
    model_creation.add(layers.Dense(10, activation="softmax"))
    return model_creation


(training_images, training_labels), (testing_images, testing_labels) = load_data()

if __name__ == "__main__":
    display_images(training_images, training_labels)

# Select a subset of the dataset for training and testing
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Build a Convolutional Neural Network (CNN) model
model = create_model()

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model on the training data
model.fit(training_images, training_labels, epochs=10, validation_split=0.2)

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save the trained model
model.save("image_classifier/model/image_classifier.keras")
