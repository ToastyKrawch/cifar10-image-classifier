"""
This module defines the Locust load testing configuration for the application.
"""

# pylint: disable=import-error
from locust import HttpUser, TaskSet, between, task


class UserBehavior(TaskSet):
    """
    Defines the behavior of a simulated user in the load test.
    """

    @task
    def upload_image(self):
        """
        Simulates a POST request to the root endpoint.
        """
        with open("image_classifier/tests/test_image.png", "rb") as image_file:
            self.client.post(
                "/", files={"file": ("test_image.png", image_file, "image/png")}
            )

    @task
    def index(self):
        """
        Simulates a GET request to the root endpoint.
        """
        self.client.get("/")


# pylint: disable=too-few-public-methods
class WebsiteUser(HttpUser):
    """
    Simulates a user interacting with the website.
    """

    tasks = [UserBehavior]
    wait_time = between(1, 5)
