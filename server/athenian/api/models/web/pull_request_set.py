from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.included_native_users import IncludedNativeUsers
from athenian.api.models.web.pull_request import PullRequest


class PullRequestSet(Model):
    """List of pull requests together with the participant profile pictures."""

    openapi_types = {
        "include": IncludedNativeUsers,
        "data": List[PullRequest],
    }

    attribute_map = {"include": "include", "data": "data"}

    __slots__ = ["_" + k for k in openapi_types]

    def __init__(
        self,
        include: Optional[IncludedNativeUsers] = None,
        data: Optional[List[PullRequest]] = None,
    ):
        """PullRequestSet - a model defined in OpenAPI

        :param include: The include of this PullRequestSet.
        :param data: The data of this PullRequestSet.
        """
        self._include = include
        self._data = data

    @property
    def include(self) -> IncludedNativeUsers:
        """Gets the include of this PullRequestSet.

        :return: The include of this PullRequestSet.
        """
        return self._include

    @include.setter
    def include(self, include: IncludedNativeUsers):
        """Sets the include of this PullRequestSet.

        :param include: The include of this PullRequestSet.
        """
        self._include = include

    @property
    def data(self) -> List[PullRequest]:
        """Gets the data of this PullRequestSet.

        List of matched pull requests.

        :return: The data of this PullRequestSet.
        """
        return self._data

    @data.setter
    def data(self, data: List[PullRequest]):
        """Sets the data of this PullRequestSet.

        List of matched pull requests.

        :param data: The data of this PullRequestSet.
        """
        self._data = data
