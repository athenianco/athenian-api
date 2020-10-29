from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.filtered_release import FilteredRelease
from athenian.api.models.web.included_jira_issues import _IncludedJIRAIssues
from athenian.api.models.web.included_native_users import _IncludedNativeUsers


FilteredReleasesInclude = AllOf(_IncludedNativeUsers, _IncludedJIRAIssues)


class FilteredReleases(Model):
    """Response of `/filter/releases` - releases metadata and user details."""

    openapi_types = {
        "include": FilteredReleasesInclude,
        "data": List[FilteredRelease],
    }

    attribute_map = {"include": "include", "data": "data"}

    def __init__(
        self,
        include: Optional[FilteredReleasesInclude] = None,
        data: Optional[List[FilteredRelease]] = None,
    ):
        """FilteredReleases - a model defined in OpenAPI

        :param include: The include of this FilteredReleases.
        :param data: The data of this FilteredReleases.
        """
        self._include = include
        self._data = data

    @property
    def include(self) -> FilteredReleasesInclude:
        """Gets the include of this FilteredReleases.

        :return: The include of this FilteredReleases.
        """
        return self._include

    @include.setter
    def include(self, include: FilteredReleasesInclude):
        """Sets the include of this FilteredReleases.

        :param include: The include of this FilteredReleases.
        """
        self._include = include

    @property
    def data(self) -> List[FilteredRelease]:
        """Gets the data of this FilteredReleases.

        Response of `/filter/releases`.

        :return: The data of this FilteredReleases.
        """
        return self._data

    @data.setter
    def data(self, data: List[FilteredRelease]):
        """Sets the data of this FilteredReleases.

        Response of `/filter/releases`.

        :param data: The data of this FilteredReleases.
        """
        self._data = data
