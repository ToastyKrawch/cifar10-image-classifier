# pylint: disable=import-error, no-name-in-module

"""
model_creation.py: This script builds, trains, and evaluates a Convolutional Neural Network (CNN)
on the CIFAR-10 dataset, and saves the trained model as `image_classifier.keras`.
"""

# Import necessary libraries
from tensorflow.keras import layers, models

from common import class_names, display_sample_images, load_cifar10_data

# Load dataset
(training_images, training_labels), (testing_images, testing_labels) = (
    load_cifar10_data()
)

# Normalize pixel values to be between 0 and 1
training_images, testing_images = training_images / 255, testing_images / 255

# Display sample images from the dataset
display_sample_images(training_images, training_labels, class_names)

# Select a subset of the dataset for training and testing
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Build a Convolutional Neural Network (CNN) model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

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
model.save("image_classifier.keras")
