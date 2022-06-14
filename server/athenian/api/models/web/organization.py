from typing import Optional

from athenian.api.models.web.base_model_ import Model


class Organization(Model):
    """GitHub organization details."""

    attribute_types = {
        "name": str,
        "avatar_url": str,
        "login": str,
    }
    attribute_map = {
        "name": "name",
        "avatar_url": "avatar_url",
        "login": "login",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        avatar_url: Optional[str] = None,
        login: Optional[str] = None,
    ):
        """Organization - a model defined in OpenAPI

        :param name: The name of this Organization.
        :param avatar_url: The avatar_url of this Organization.
        :param login: The login of this Organization.
        """
        self._name = name
        self._avatar_url = avatar_url
        self._login = login

    @property
    def name(self) -> str:
        """Gets the name of this Organization.

        :return: The name of this Organization.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this Organization.

        :param name: The name of this Organization.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def avatar_url(self) -> str:
        """Gets the avatar_url of this Organization.

        :return: The avatar_url of this Organization.
        """
        return self._avatar_url

    @avatar_url.setter
    def avatar_url(self, avatar_url: str):
        """Sets the avatar_url of this Organization.

        :param avatar_url: The avatar_url of this Organization.
        """
        if avatar_url is None:
            raise ValueError("Invalid value for `avatar_url`, must not be `None`")

        self._avatar_url = avatar_url

    @property
    def login(self) -> str:
        """Gets the login of this Organization.

        :return: The login of this Organization.
        """
        return self._login

    @login.setter
    def login(self, login: str):
        """Sets the login of this Organization.

        :param login: The login of this Organization.
        """
        if login is None:
            raise ValueError("Invalid value for `login`, must not be `None`")

        self._login = login
