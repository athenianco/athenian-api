from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.developer_updates import DeveloperUpdates


class DeveloperSummary(Model):
    """Developer activity statistics and profile details."""

    openapi_types = {
        "login": str,
        "name": str,
        "avatar": str,
        "updates": DeveloperUpdates,
    }

    attribute_map = {
        "login": "login",
        "name": "name",
        "avatar": "avatar",
        "updates": "updates",
    }

    __slots__ = ["_" + k for k in openapi_types]

    def __init__(
        self,
        login: str = None,
        name: str = None,
        avatar: str = None,
        updates: DeveloperUpdates = None,
    ):
        """DeveloperSummary - a model defined in OpenAPI

        :param login: The login of this DeveloperSummary.
        :param name: The name of this DeveloperSummary.
        :param avatar: The avatar of this DeveloperSummary.
        :param updates: The updates of this DeveloperSummary.
        """
        self._login = login
        self._name = name
        self._avatar = avatar
        self._updates = updates

    @property
    def login(self) -> str:
        """Gets the login of this DeveloperSummary.

        Developer's login name.

        :return: The login of this DeveloperSummary.
        """
        return self._login

    @login.setter
    def login(self, login: str):
        """Sets the login of this DeveloperSummary.

        Developer's login name.

        :param login: The login of this DeveloperSummary.
        """
        if login is None:
            raise ValueError("Invalid value for `login`, must not be `None`")

        self._login = login

    @property
    def name(self) -> str:
        """Gets the name of this DeveloperSummary.

        Developer's full name.

        :return: The name of this DeveloperSummary.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this DeveloperSummary.

        Developer's full name.

        :param name: The name of this DeveloperSummary.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def avatar(self) -> str:
        """Gets the avatar of this DeveloperSummary.

        Developer's avatar URL.

        :return: The avatar of this DeveloperSummary.
        """
        return self._avatar

    @avatar.setter
    def avatar(self, avatar: str):
        """Sets the avatar of this DeveloperSummary.

        Developer's avatar URL.

        :param avatar: The avatar of this DeveloperSummary.
        """
        if avatar is None:
            raise ValueError("Invalid value for `avatar`, must not be `None`")

        self._avatar = avatar

    @property
    def updates(self) -> DeveloperUpdates:
        """Gets the updates of this DeveloperSummary.

        :return: The updates of this DeveloperSummary.
        """
        return self._updates

    @updates.setter
    def updates(self, updates: DeveloperUpdates):
        """Sets the updates of this DeveloperSummary.

        :param updates: The updates of this DeveloperSummary.
        """
        if updates is None:
            raise ValueError("Invalid value for `updates`, must not be `None`")

        self._updates = updates
