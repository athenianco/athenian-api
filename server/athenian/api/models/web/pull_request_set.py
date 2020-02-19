from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.pull_request import PullRequest
from athenian.api.models.web.pull_request_set_include import PullRequestSetInclude


class PullRequestSet(Model):
    """List of pull requests together with the participant profile pictures."""

    def __init__(
        self,
        include: Optional[PullRequestSetInclude] = None,
        data: Optional[List[PullRequest]] = None,
    ):
        """PullRequestSet - a model defined in OpenAPI

        :param include: The include of this PullRequestSet.
        :param data: The data of this PullRequestSet.
        """
        self.openapi_types = {
            "include": PullRequestSetInclude,
            "data": List[PullRequest],
        }

        self.attribute_map = {"include": "include", "data": "data"}

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
