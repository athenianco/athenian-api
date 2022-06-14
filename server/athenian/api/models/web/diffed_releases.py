from typing import Dict, List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.release_diff import ReleaseDiff
from athenian.api.models.web.release_set import ReleaseSetInclude


class DiffedReleases(Model):
    """Response of `/diff/releases` - the found inner releases for each repository."""

    attribute_types = {
        "include": ReleaseSetInclude,
        "data": Dict[str, List[ReleaseDiff]],
    }

    attribute_map = {"include": "include", "data": "data"}

    def __init__(
        self,
        include: Optional[ReleaseSetInclude] = None,
        data: Optional[Dict[str, List[ReleaseDiff]]] = None,
    ):
        """DiffedReleases - a model defined in OpenAPI

        :param include: The include of this DiffedReleases.
        :param data: The data of this DiffedReleases.
        """
        self._include = include
        self._data = data

    @property
    def include(self) -> ReleaseSetInclude:
        """Gets the include of this DiffedReleases.

        :return: The include of this DiffedReleases.
        """
        return self._include

    @include.setter
    def include(self, include: ReleaseSetInclude):
        """Sets the include of this DiffedReleases.

        :param include: The include of this DiffedReleases.
        """
        if include is None:
            raise ValueError("Invalid value for `include`, must not be `None`")

        self._include = include

    @property
    def data(self) -> Dict[str, List[ReleaseDiff]]:
        """Gets the data of this DiffedReleases.

        Mapping from repository names to diff results.

        :return: The data of this DiffedReleases.
        """
        return self._data

    @data.setter
    def data(self, data: Dict[str, List[ReleaseDiff]]):
        """Sets the data of this DiffedReleases.

        Mapping from repository names to diff results.

        :param data: The data of this DiffedReleases.
        """
        if data is None:
            raise ValueError("Invalid value for `data`, must not be `None`")

        self._data = data
