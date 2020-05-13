from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class TeamUpdateRequest(Model):
    """Team update request."""

    def __init__(
            self,
            id: Optional[int] = None,
            name: Optional[str] = None,
            members: Optional[List[str]] = None,
    ):
        """TeamUpdateRequest - a model defined in OpenAPI

        :param id: The id of this TeamUpdateRequest.
        :param name: The name of this TeamUpdateRequest.
        :param members: The members of this TeamUpdateRequest.
        """
        self.openapi_types = {"id": int, "name": str, "members": List[str]}

        self.attribute_map = {"id": "id", "name": "name", "members": "members"}

        self._id = id
        self._name = name
        self._members = members

    @property
    def id(self) -> int:
        """Gets the id of this TeamUpdateRequest.

        Team identifier.

        :return: The id of this TeamUpdateRequest.
        """
        return self._id

    @id.setter
    def id(self, id: int):
        """Sets the id of this TeamUpdateRequest.

        Team identifier.

        :param id: The id of this TeamUpdateRequest.
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")

        self._id = id

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
