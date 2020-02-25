from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.commit_group_for_filter_commits_request import \
    CommitGroupForFilterCommitsRequest
from athenian.api.models.web.included_native_users import IncludedNativeUsers


class CommitsGroupedByTime(Model):
    """Lists of commits for each time interval."""

    def __init__(
        self,
        data: Optional[List[CommitGroupForFilterCommitsRequest]] = None,
        include: Optional[IncludedNativeUsers] = None,
    ):
        """CommitsGroupedByTime - a model defined in OpenAPI

        :param data: The data of this CommitsGroupedByTime.
        :param include: The include of this CommitsGroupedByTime.
        """
        self.openapi_types = {
            "data": List[CommitGroupForFilterCommitsRequest],
            "include": IncludedNativeUsers,
        }

        self.attribute_map = {"data": "data", "include": "include"}

        self._data = data
        self._include = include

    @property
    def data(self) -> List[CommitGroupForFilterCommitsRequest]:
        """Gets the data of this CommitsGroupedByTime.

        :return: The data of this CommitsGroupedByTime.
        """
        return self._data

    @data.setter
    def data(self, data: List[CommitGroupForFilterCommitsRequest]):
        """Sets the data of this CommitsGroupedByTime.

        :param data: The data of this CommitsGroupedByTime.
        """
        if data is None:
            raise ValueError("Invalid value for `data`, must not be `None`")

        self._data = data

    @property
    def include(self) -> IncludedNativeUsers:
        """Gets the include of this CommitsGroupedByTime.

        :return: The include of this CommitsGroupedByTime.
        """
        return self._include

    @include.setter
    def include(self, include: IncludedNativeUsers):
        """Sets the include of this CommitsGroupedByTime.

        :param include: The include of this CommitsGroupedByTime.
        """
        if include is None:
            raise ValueError("Invalid value for `include`, must not be `None`")

        self._include = include
