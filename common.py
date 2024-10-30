# pylint: disable=import-error, no-name-in-module

"""
common.py: Contains common functionality for loading the CIFAR-10 dataset and defining
class names used in multiple scripts.
"""

# common.py: Common functionality for CIFAR-10 dataset handling
import matplotlib.pyplot as plt
from tensorflow.keras import datasets


# Load CIFAR-10 dataset and normalize pixel values
def load_cifar10_data():
    """
    Load and normalize the CIFAR-10 dataset.

    Returns:
        tuple: Tuple containing training and testing images and labels.
    """
    (training_images, training_labels), (testing_images, testing_labels) = (
        datasets.cifar10.load_data()
    )
    training_images, testing_images = training_images / 255, testing_images / 255
    return (training_images, training_labels), (testing_images, testing_labels)


def display_sample_images(images, labels, classes):
    """
    Display sample images from the dataset.

    Args:
        images (numpy.ndarray): The images to display.
        labels (numpy.ndarray): The corresponding labels for the images.
        classes (list): The class names corresponding to the labels.
    """
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i], cmap=plt.cm.binary)  # pylint: disable=no-member
        plt.xlabel(classes[labels[i][0]])
    plt.show()


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
