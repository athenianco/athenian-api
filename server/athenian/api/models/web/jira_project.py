from datetime import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAProject(Model):
    """JIRA project setting."""

    openapi_types = {
        "name": str,
        "key": str,
        "avatar_url": str,
        "enabled": bool,
        "issues_count": int,
        "last_update": datetime,
    }

    attribute_map = {
        "name": "name",
        "key": "key",
        "avatar_url": "avatar_url",
        "enabled": "enabled",
        "issues_count": "issues_count",
        "last_update": "last_update",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        key: Optional[str] = None,
        avatar_url: Optional[str] = None,
        enabled: Optional[bool] = None,
        issues_count: Optional[int] = None,
        last_update: Optional[datetime] = None,
    ):
        """JIRAProject - a model defined in OpenAPI

        :param name: The name of this JIRAProject.
        :param key: The key of this JIRAProject.
        :param avatar_url: The avatar_url of this JIRAProject.
        :param enabled: The enabled of this JIRAProject.
        :param issues_count: The issues_count of this JIRAProject.
        :param last_update: The last_update of this JIRAProject.
        """
        self._name = name
        self._key = key
        self._avatar_url = avatar_url
        self._enabled = enabled
        self._issues_count = issues_count
        self._last_update = last_update

    @property
    def name(self) -> str:
        """Gets the name of this JIRAProject.

        Long name of the project.

        :return: The name of this JIRAProject.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this JIRAProject.

        Long name of the project.

        :param name: The name of this JIRAProject.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def key(self) -> str:
        """Gets the key of this JIRAProject.

        Short prefix of the project.

        :return: The key of this JIRAProject.
        """
        return self._key

    @key.setter
    def key(self, key: str):
        """Sets the key of this JIRAProject.

        Short prefix of the project.

        :param key: The key of this JIRAProject.
        """
        if key is None:
            raise ValueError("Invalid value for `key`, must not be `None`")

        self._key = key

    @property
    def avatar_url(self) -> str:
        """Gets the avatar_url of this JIRAProject.

        Avatar URL of the project.

        :return: The avatar_url of this JIRAProject.
        """
        return self._avatar_url

    @avatar_url.setter
    def avatar_url(self, avatar_url: str):
        """Sets the avatar_url of this JIRAProject.

        Avatar URL of the project.

        :param avatar_url: The avatar_url of this JIRAProject.
        """
        if avatar_url is None:
            raise ValueError("Invalid value for `avatar_url`, must not be `None`")

        self._avatar_url = avatar_url

    @property
    def enabled(self) -> bool:
        """Gets the enabled of this JIRAProject.

        Indicates whether this project is enabled for analysis.

        :return: The enabled of this JIRAProject.
        """
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: bool):
        """Sets the enabled of this JIRAProject.

        Indicates whether this project is enabled for analysis.

        :param enabled: The enabled of this JIRAProject.
        """
        if enabled is None:
            raise ValueError("Invalid value for `enabled`, must not be `None`")

        self._enabled = enabled

    @property
    def issues_count(self) -> int:
        """Gets the issues_count of this JIRAProject.

        Avatar URL of the project.

        :return: The issues_count of this JIRAProject.
        """
        return self._issues_count

    @issues_count.setter
    def issues_count(self, issues_count: int):
        """Sets the issues_count of this JIRAProject.

        Avatar URL of the project.

        :param issues_count: The issues_count of this JIRAProject.
        """
        if issues_count is None:
            raise ValueError("Invalid value for `issues_count`, must not be `None`")

        self._issues_count = issues_count

    @property
    def last_update(self) -> Optional[datetime]:
        """Gets the last_update of this JIRAProject.

        Avatar URL of the project.

        :return: The last_update of this JIRAProject.
        """
        return self._last_update

    @last_update.setter
    def last_update(self, last_update: Optional[datetime]):
        """Sets the last_update of this JIRAProject.

        Avatar URL of the project.

        :param last_update: The last_update of this JIRAProject.
        """
        self._last_update = last_update
