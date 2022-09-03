from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.commit import Commit
from athenian.api.models.web.included_deployments import _IncludedDeployments
from athenian.api.models.web.included_native_users import _IncludedNativeUsers

CommitsListInclude = AllOf(
    _IncludedNativeUsers, _IncludedDeployments, name="CommitsListInclude", module=__name__,
)


class CommitsList(Model):
    """Lists of commits for each time interval."""

    attribute_types = {
        "data": List[Commit],
        "include": CommitsListInclude,
    }

    attribute_map = {"data": "data", "include": "include"}

    def __init__(
        self,
        data: Optional[List[Commit]] = None,
        include: Optional[CommitsListInclude] = None,
    ):
        """CommitsList - a model defined in OpenAPI

        :param data: The data of this CommitsList.
        :param include: The include of this CommitsList.
        """
        self._data = data
        self._include = include

    @property
    def data(self) -> List[Commit]:
        """Gets the data of this CommitsList.

        :return: The data of this CommitsList.
        """
        return self._data

    @data.setter
    def data(self, data: List[Commit]):
        """Sets the data of this CommitsList.

        :param data: The data of this CommitsList.
        """
        if data is None:
            raise ValueError("Invalid value for `data`, must not be `None`")

        self._data = data

    @property
    def include(self) -> CommitsListInclude:
        """Gets the include of this CommitsList.

        :return: The include of this CommitsList.
        """
        return self._include

    @include.setter
    def include(self, include: CommitsListInclude):
        """Sets the include of this CommitsList.

        :param include: The include of this CommitsList.
        """
        if include is None:
            raise ValueError("Invalid value for `include`, must not be `None`")

        self._include = include
