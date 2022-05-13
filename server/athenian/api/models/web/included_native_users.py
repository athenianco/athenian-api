from typing import Dict, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.included_native_user import IncludedNativeUser


class _IncludedNativeUsers(Model, sealed=False):
    """Mapping user login -> user details.

    The users are mentioned in PRs in "PullRequestSet.data".
    """

    attribute_types = {"users": Dict[str, IncludedNativeUser]}
    attribute_map = {"users": "users"}

    def __init__(self, users: Optional[Dict[str, IncludedNativeUser]] = None):
        """IncludedNativeUsers - a model defined in OpenAPI

        :param users: The users of this IncludedNativeUsers.
        """
        self._users = users

    @property
    def users(self) -> Dict[str, IncludedNativeUser]:
        """Gets the users of this IncludedNativeUsers.

        Mapping user login -> user details. The users are mentioned in PRs in \"data\".

        :return: The users of this IncludedNativeUsers.
        """
        return self._users

    @users.setter
    def users(self, users: Dict[str, IncludedNativeUser]):
        """Sets the users of this IncludedNativeUsers.

        Mapping user login -> user details. The users are mentioned in PRs in \"data\".

        :param users: The users of this IncludedNativeUsers.
        """
        if users is None:
            raise ValueError("Invalid value for `users`, must not be `None`")

        self._users = users


class IncludedNativeUsers(_IncludedNativeUsers):
    """Mapping user login -> user details.

    The users are mentioned in PRs in "PullRequestSet.data".
    """
