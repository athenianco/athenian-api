from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.filtered_release import FilteredRelease


class ReleaseDiff(Model):
    """Inner releases between `old` and `new`, including the latter."""

    attribute_types = {"old": str, "new": str, "releases": List[FilteredRelease]}
    attribute_map = {"old": "old", "new": "new", "releases": "releases"}

    def __init__(
        self,
        old: Optional[str] = None,
        new: Optional[str] = None,
        releases: Optional[List[FilteredRelease]] = None,
    ):
        """ReleaseDiff - a model defined in OpenAPI

        :param old: The old of this ReleaseDiff.
        :param new: The new of this ReleaseDiff.
        :param releases: The releases of this ReleaseDiff.
        """
        self._old = old
        self._new = new
        self._releases = releases

    @property
    def old(self) -> str:
        """Gets the old of this ReleaseDiff.

        :return: The old of this ReleaseDiff.
        """
        return self._old

    @old.setter
    def old(self, old: str):
        """Sets the old of this ReleaseDiff.

        :param old: The old of this ReleaseDiff.
        """
        if old is None:
            raise ValueError("Invalid value for `old`, must not be `None`")

        self._old = old

    @property
    def new(self) -> str:
        """Gets the new of this ReleaseDiff.

        :return: The new of this ReleaseDiff.
        """
        return self._new

    @new.setter
    def new(self, new: str):
        """Sets the new of this ReleaseDiff.

        :param new: The new of this ReleaseDiff.
        """
        if new is None:
            raise ValueError("Invalid value for `new`, must not be `None`")

        self._new = new

    @property
    def releases(self) -> List[FilteredRelease]:
        """Gets the releases of this ReleaseDiff.

        List of matching release metadata.

        :return: The releases of this ReleaseDiff.
        """
        return self._releases

    @releases.setter
    def releases(self, releases: List[FilteredRelease]):
        """Sets the releases of this ReleaseDiff.

        List of matching release metadata.

        :param releases: The releases of this ReleaseDiff.
        """
        if releases is None:
            raise ValueError("Invalid value for `releases`, must not be `None`")

        self._releases = releases
