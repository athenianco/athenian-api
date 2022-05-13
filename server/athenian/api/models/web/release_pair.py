from typing import Optional

from athenian.api.models.web.base_model_ import Model


class ReleasePair(Model):
    """A pair of release names within the same repository."""

    attribute_types = {"old": str, "new": str}
    attribute_map = {"old": "old", "new": "new"}

    def __init__(self, old: Optional[str] = None, new: Optional[str] = None):
        """ReleasePair - a model defined in OpenAPI

        :param old: The old of this ReleasePair.
        :param new: The new of this ReleasePair.
        """
        self._old = old
        self._new = new

    @property
    def old(self) -> str:
        """Gets the old of this ReleasePair.

        Older release name.

        :return: The old of this ReleasePair.
        """
        return self._old

    @old.setter
    def old(self, old: str):
        """Sets the old of this ReleasePair.

        Older release name.

        :param old: The old of this ReleasePair.
        """
        if old is None:
            raise ValueError("Invalid value for `old`, must not be `None`")

        self._old = old

    @property
    def new(self) -> str:
        """Gets the new of this ReleasePair.

        Newer release name.

        :return: The new of this ReleasePair.
        """
        return self._new

    @new.setter
    def new(self, new: str):
        """Sets the new of this ReleasePair.

        Newer release name.

        :param new: The new of this ReleasePair.
        """
        if new is None:
            raise ValueError("Invalid value for `new`, must not be `None`")

        self._new = new
