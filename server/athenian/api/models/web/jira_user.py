from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAUser(Model):
    """Details about a JIRA user."""

    attribute_types = {
        "name": str,
        "avatar": str,
        "type": str,
        "developer": Optional[str],
    }
    attribute_map = {
        "name": "name",
        "avatar": "avatar",
        "type": "type",
        "developer": "developer",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        avatar: Optional[str] = None,
        type: Optional[str] = None,
        developer: Optional[str] = None,
    ):
        """JIRAUser - a model defined in OpenAPI

        :param name: The name of this JIRAUser.
        :param avatar: The avatar of this JIRAUser.
        :param type: The type of this JIRAUser.
        :param developer: The developer of this JIRAUser.
        """
        self._name = name
        self._avatar = avatar
        self._type = type
        self._developer = developer

    @property
    def name(self) -> str:
        """Gets the name of this JIRAUser.

        Full name of the user.

        :return: The name of this JIRAUser.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this JIRAUser.

        Full name of the user.

        :param name: The name of this JIRAUser.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def avatar(self) -> str:
        """Gets the avatar of this JIRAUser.

        User's profile picture URL.

        :return: The avatar of this JIRAUser.
        """
        return self._avatar

    @avatar.setter
    def avatar(self, avatar: str):
        """Sets the avatar of this JIRAUser.

        User's profile picture URL.

        :param avatar: The avatar of this JIRAUser.
        """
        if avatar is None:
            raise ValueError("Invalid value for `avatar`, must not be `None`")

        self._avatar = avatar

    @property
    def type(self) -> str:
        """Gets the type of this JIRAUser.

        * `atlassian` indicates a regular account backed by a human.
        * `app` indicates a service account.
        * `customer` indicates an external service desk account.

        :return: The type of this JIRAUser.
        """
        return self._type

    @type.setter
    def type(self, type: str):
        """Sets the type of this JIRAUser.

        * `atlassian` indicates a regular account backed by a human.
        * `app` indicates a service account.
        * `customer` indicates an external service desk account.

        :param type: The type of this JIRAUser.
        """
        allowed_values = {"atlassian", "app", "customer"}
        if type not in allowed_values:
            raise ValueError(
                "Invalid value for `type` (%s), must be one of %s" % (type, allowed_values),
            )

        self._type = type

    @property
    def developer(self) -> Optional[str]:
        """Gets the developer of this JIRAUser.

        Mapped developer identity.

        :return: The developer of this JIRAUser.
        """
        return self._developer

    @developer.setter
    def developer(self, developer: Optional[str]):
        """Sets the developer of this JIRAUser.

        Mapped developer identity.

        :param developer: The developer of this JIRAUser.
        """
        self._developer = developer
