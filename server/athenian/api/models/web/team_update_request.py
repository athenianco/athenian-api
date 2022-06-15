from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class TeamUpdateRequest(Model):
    """Team update request."""

    attribute_types = {
        "name": str,
        "members": List[str],
        "parent": int,
    }
    attribute_map = {
        "name": "name",
        "members": "members",
        "parent": "parent",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        members: Optional[List[str]] = None,
        parent: Optional[int] = None,
    ):
        """TeamUpdateRequest - a model defined in OpenAPI

        :param name: The name of this TeamUpdateRequest.
        :param members: The members of this TeamUpdateRequest.
        :param parent: The parent of this TeamUpdateRequest.
        """
        self._name = name
        self._members = members
        self._parent = parent

    @property
    def name(self) -> str:
        """Gets the name of this TeamUpdateRequest.

        Name of the team.

        :return: The name of this TeamUpdateRequest.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this TeamUpdateRequest.

        Name of the team.

        :param name: The name of this TeamUpdateRequest.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def members(self) -> List[str]:
        """Gets the members of this TeamUpdateRequest.

        A set of developers. An empty list disables the filter and includes everybody.
        Duplicates are automatically ignored.

        :return: The members of this TeamUpdateRequest.
        """
        return self._members

    @members.setter
    def members(self, members: List[str]):
        """Sets the members of this TeamUpdateRequest.

        A set of developers. An empty list disables the filter and includes everybody.
        Duplicates are automatically ignored.

        :param members: The members of this TeamUpdateRequest.
        """
        if members is None:
            raise ValueError("Invalid value for `members`, must not be `None`")

        self._members = members

    @property
    def parent(self) -> Optional[int]:
        """Gets the parent of this TeamUpdateRequest.

        New identifier of the higher level team.

        :return: The parent of this TeamUpdateRequest.
        """
        return self._parent

    @parent.setter
    def parent(self, parent: Optional[int]):
        """Sets the parent of this TeamUpdateRequest.

        New identifier of the higher level team.

        :param parent: The parent of this TeamUpdateRequest.
        """
        self._parent = parent
