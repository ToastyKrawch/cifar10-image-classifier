# pylint: disable=no-member import-error no-name-in-module

"""
images.py: This script displays images
"""

import matplotlib.pyplot as plt
from app import class_names
from tensorflow.keras import datasets  # type: ignore


# if __name__ == "__main__":
# Display sample images from the dataset
def display_images(tr_images, tr_labels):
    """
    displays images
    """
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(tr_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[tr_labels[i][0]])
    plt.show()


# Load CIFAR-10 dataset
def load_data():
    """
    loads data
    """
    (training_images, training_labels), (testing_images, testing_labels) = (
        datasets.cifar10.load_data()
    )

    training_images, testing_images = training_images / 255, testing_images / 255

    return (training_images, training_labels), (testing_images, testing_labels)
