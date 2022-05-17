from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.included_deployments import _IncludedDeployments
from athenian.api.models.web.included_native_users import _IncludedNativeUsers
from athenian.api.models.web.pull_request import PullRequest


PullRequestSetInclude = AllOf(_IncludedNativeUsers, _IncludedDeployments,
                              name="PullRequestSetInclude", module=__name__)


class PullRequestSet(Model):
    """List of pull requests together with the participant profile pictures."""

    attribute_types = {
        "include": PullRequestSetInclude,
        "data": List[PullRequest],
    }

    attribute_map = {"include": "include", "data": "data"}

    def __init__(
        self,
        include: Optional[PullRequestSetInclude] = None,
        data: Optional[List[PullRequest]] = None,
    ):
        """PullRequestSet - a model defined in OpenAPI

        :param include: The include of this PullRequestSet.
        :param data: The data of this PullRequestSet.
        """
        self._include = include
        self._data = data

    @property
    def include(self) -> PullRequestSetInclude:
        """Gets the include of this PullRequestSet.

        :return: The include of this PullRequestSet.
        """
        return self._include

    @include.setter
    def include(self, include: PullRequestSetInclude):
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
