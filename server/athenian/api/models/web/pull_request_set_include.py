from typing import Dict, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.pull_request_set_include_user import PullRequestSetIncludeUser


class PullRequestSetInclude(Model):
    """Mapping user login -> user details.

    The users are mentioned in PRs in "PullRequestSet.data".
    """

    def __init__(self, users: Optional[Dict[str, PullRequestSetIncludeUser]] = None):
        """PullRequestSetInclude - a model defined in OpenAPI

        :param users: The users of this PullRequestSetInclude.
        """
        self.openapi_types = {"users": Dict[str, PullRequestSetIncludeUser]}

        self.attribute_map = {"users": "users"}

        self._users = users

    @property
    def users(self) -> Dict[str, PullRequestSetIncludeUser]:
        """Gets the users of this PullRequestSetInclude.

        Mapping user login -> user details. The users are mentioned in PRs in \"data\".

        :return: The users of this PullRequestSetInclude.
        """
        return self._users

    @users.setter
    def users(self, users: Dict[str, PullRequestSetIncludeUser]):
        """Sets the users of this PullRequestSetInclude.

        Mapping user login -> user details. The users are mentioned in PRs in \"data\".

        :param users: The users of this PullRequestSetInclude.
        """
        if users is None:
            raise ValueError("Invalid value for `users`, must not be `None`")

        self._users = users
