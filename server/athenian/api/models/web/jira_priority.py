from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAPriority(Model):
    """JIRA issue priority details."""

    openapi_types = {"name": str, "image": str, "value": int}
    attribute_map = {"name": "name", "image": "image", "value": "value"}

    def __init__(self,
                 name: Optional[str] = None,
                 image: Optional[str] = None,
                 value: Optional[int] = None):
        """JIRAPriority - a model defined in OpenAPI

        :param name: The name of this JIRAPriority.
        :param image: The image of this JIRAPriority.
        :param value: The value of this JIRAPriority.
        """
        self._name = name
        self._image = image
        self._value = value

    @property
    def name(self) -> str:
        """Gets the name of this JIRAPriority.

        Name of the priority.

        :return: The name of this JIRAPriority.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this JIRAPriority.

        Name of the priority.

        :param name: The name of this JIRAPriority.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def image(self) -> str:
        """Gets the image of this JIRAPriority.

        URL of the picture that indicates the priority.

        :return: The image of this JIRAPriority.
        """
        return self._image

    @image.setter
    def image(self, image: str):
        """Sets the image of this JIRAPriority.

        URL of the picture that indicates the priority.

        :param image: The image of this JIRAPriority.
        """
        if image is None:
            raise ValueError("Invalid value for `image`, must not be `None`")

        self._image = image

    @property
    def value(self) -> int:
        """Gets the value of this JIRAPriority.

        Measure of importance (bigger is more important).

        :return: The value of this JIRAPriority.
        """
        return self._value

    @value.setter
    def value(self, value: int):
        """Sets the value of this JIRAPriority.

        Measure of importance (bigger is more important).

        :param value: The value of this JIRAPriority.
        """
        if value is None:
            raise ValueError("Invalid value for `value`, must not be `None`")
        if value is not None and value < 0:
            raise ValueError(
                "Invalid value for `value`, must be a value greater than or equal to `0`")

        self._value = value
