from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.filtered_release import FilteredRelease
from athenian.api.models.web.included_deployments import _IncludedDeployments
from athenian.api.models.web.included_jira_issues import _IncludedJIRAIssues
from athenian.api.models.web.included_native_users import _IncludedNativeUsers


ReleaseSetInclude = AllOf(_IncludedNativeUsers, _IncludedJIRAIssues, _IncludedDeployments,
                          name="ReleaseSetInclude", module=__name__)


class ReleaseSet(Model):
    """Release metadata and contributor user details."""

    attribute_types = {
        "include": ReleaseSetInclude,
        "data": List[FilteredRelease],
    }

    attribute_map = {"include": "include", "data": "data"}

    def __init__(
        self,
        include: Optional[ReleaseSetInclude] = None,
        data: Optional[List[FilteredRelease]] = None,
    ):
        """ReleaseSet - a model defined in OpenAPI

        :param include: The include of this ReleaseSet.
        :param data: The data of this ReleaseSet.
        """
        self._include = include
        self._data = data

    @property
    def include(self) -> ReleaseSetInclude:
        """Gets the include of this ReleaseSet.

        :return: The include of this ReleaseSet.
        """
        return self._include

    @include.setter
    def include(self, include: ReleaseSetInclude):
        """Sets the include of this ReleaseSet.

        :param include: The include of this ReleaseSet.
        """
        self._include = include

    @property
    def data(self) -> List[FilteredRelease]:
        """Gets the data of this ReleaseSet.

        Response of `/filter/releases`.

        :return: The data of this ReleaseSet.
        """
        return self._data

    @data.setter
    def data(self, data: List[FilteredRelease]):
        """Sets the data of this ReleaseSet.

        Response of `/filter/releases`.

        :param data: The data of this ReleaseSet.
        """
        self._data = data
