from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.developer_updates import DeveloperUpdates


class DeveloperSummary(Model):
    """Developer activity statistics and profile details."""

    attribute_types = {
        "login": str,
        "name": str,
        "avatar": str,
        "updates": DeveloperUpdates,
        "jira_user": Optional[str],
    }

    attribute_map = {
        "login": "login",
        "name": "name",
        "avatar": "avatar",
        "updates": "updates",
        "jira_user": "jira_user",
    }

    def __init__(
        self,
        login: Optional[str] = None,
        name: Optional[str] = None,
        avatar: Optional[str] = None,
        updates: Optional[DeveloperUpdates] = None,
        jira_user: Optional[str] = None,
    ):
        """DeveloperSummary - a model defined in OpenAPI

        :param login: The login of this DeveloperSummary.
        :param name: The name of this DeveloperSummary.
        :param avatar: The avatar of this DeveloperSummary.
        :param updates: The updates of this DeveloperSummary.
        :param jira_user: The jira_user of this DeveloperSummary.
        """
        self._login = login
        self._name = name
        self._avatar = avatar
        self._updates = updates
        self._jira_user = jira_user

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

    @property
    def jira_user(self) -> Optional[str]:
        """Gets the jira_user of this DeveloperSummary.

        Mapped JIRA user name.

        :return: The jira_user of this DeveloperSummary.
        """
        return self._jira_user

    @jira_user.setter
    def jira_user(self, jira_user: Optional[str]):
        """Sets the jira_user of this DeveloperSummary.

        Mapped JIRA user name.

        :param jira_user: The jira_user of this DeveloperSummary.
        """
        self._jira_user = jira_user
