from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAEpicSetting(Model):
    """Details about a JIRA issue type that we consider epic."""

    openapi_types = {
        "name": str,
        "description": str,
        "normalized_name": str,
        "icon": str,
    }

    attribute_map = {
        "name": "name",
        "description": "description",
        "normalized_name": "normalized_name",
        "icon": "icon",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        normalized_name: Optional[str] = None,
        icon: Optional[str] = None,
    ):
        """JIRAEpicSetting - a model defined in OpenAPI

        :param name: The name of this JIRAEpicSetting.
        :param description: The description of this JIRAEpicSetting.
        :param normalized_name: The normalized_name of this JIRAEpicSetting.
        :param icon: The icon of this JIRAEpicSetting.
        """
        self._name = name
        self._description = description
        self._normalized_name = normalized_name
        self._icon = icon

    def __lt__(self, other: "JIRAEpicSetting") -> bool:
        """Support sorting."""
        return self.normalized_name < other.normalized_name

    @property
    def name(self) -> str:
        """Gets the name of this JIRAEpicSetting.

        :return: The name of this JIRAEpicSetting.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this JIRAEpicSetting.

        :param name: The name of this JIRAEpicSetting.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def description(self) -> str:
        """Gets the description of this JIRAEpicSetting.

        :return: The description of this JIRAEpicSetting.
        """
        return self._description

    @description.setter
    def description(self, description: str):
        """Sets the description of this JIRAEpicSetting.

        :param description: The description of this JIRAEpicSetting.
        """
        if description is None:
            raise ValueError("Invalid value for `description`, must not be `None`")

        self._description = description

    @property
    def normalized_name(self) -> str:
        """Gets the normalized_name of this JIRAEpicSetting.

        :return: The normalized_name of this JIRAEpicSetting.
        """
        return self._normalized_name

    @normalized_name.setter
    def normalized_name(self, normalized_name: str):
        """Sets the normalized_name of this JIRAEpicSetting.

        :param normalized_name: The normalized_name of this JIRAEpicSetting.
        """
        if normalized_name is None:
            raise ValueError("Invalid value for `normalized_name`, must not be `None`")

        self._normalized_name = normalized_name

    @property
    def icon(self) -> str:
        """Gets the icon of this JIRAEpicSetting.

        :return: The icon of this JIRAEpicSetting.
        """
        return self._icon

    @icon.setter
    def icon(self, icon: str):
        """Sets the icon of this JIRAEpicSetting.

        :param icon: The icon of this JIRAEpicSetting.
        """
        if icon is None:
            raise ValueError("Invalid value for `icon`, must not be `None`")

        self._icon = icon
