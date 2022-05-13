from typing import Optional

from athenian.api.models.web.base_model_ import Model


class PullRequestLabel(Model):
    """Pull request label."""

    attribute_types = {"name": str, "description": str, "color": str}

    attribute_map = {
        "name": "name",
        "description": "description",
        "color": "color",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        color: Optional[str] = None,
    ):
        """PullRequestLabel - a model defined in OpenAPI

        :param name: The name of this PullRequestLabel.
        :param description: The description of this PullRequestLabel.
        :param color: The color of this PullRequestLabel.
        """
        self._name = name
        self._description = description
        self._color = color

    @property
    def name(self) -> str:
        """Gets the name of this PullRequestLabel.

        :return: The name of this PullRequestLabel.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this PullRequestLabel.

        :param name: The name of this PullRequestLabel.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def description(self) -> str:
        """Gets the description of this PullRequestLabel.

        :return: The description of this PullRequestLabel.
        """
        return self._description

    @description.setter
    def description(self, description: str):
        """Sets the description of this PullRequestLabel.

        :param description: The description of this PullRequestLabel.
        """
        self._description = description

    @property
    def color(self) -> str:
        """Gets the color of this PullRequestLabel.

        :return: The color of this PullRequestLabel.
        """
        return self._color

    @color.setter
    def color(self, color: str):
        """Sets the color of this PullRequestLabel.

        :param color: The color of this PullRequestLabel.
        """
        if color is None:
            raise ValueError("Invalid value for `color`, must not be `None`")

        self._color = color
