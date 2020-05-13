from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.contributor import Contributor


class Team(Model):
    """List of team members."""

    def __init__(
            self,
            id: Optional[int] = None,
            name: Optional[str] = None,
            members: Optional[List[Contributor]] = None,
    ):
        """Team - a model defined in OpenAPI

        :param id: The id of this Team.
        :param name: The name of this Team.
        :param members: The members of this Team.
        """
        self.openapi_types = {"id": int, "name": str, "members": List[Contributor]}

        self.attribute_map = {"id": "id", "name": "name", "members": "members"}

        self._id = id
        self._name = name
        self._members = members

    @property
    def id(self) -> int:
        """Gets the id of this Team.

        Team identifier.

        :return: The id of this Team.
        """
        return self._id

    @id.setter
    def id(self, id: int):
        """Sets the id of this Team.

        Team identifier.

        :param id: The id of this Team.
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")

        self._id = id

    @property
    def name(self) -> str:
        """Gets the name of this Team.

        Name of the team.

        :return: The name of this Team.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this Team.

        Name of the team.

        :param name: The name of this Team.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def members(self) -> List[Contributor]:
        """Gets the members of this Team.

        List of contributors.

        :return: The members of this Team.
        """
        return self._members

    @members.setter
    def members(self, members: List[Contributor]):
        """Sets the members of this Team.

        List of contributors.

        :param members: The members of this Team.
        """
        if members is None:
            raise ValueError("Invalid value for `members`, must not be `None`")

        self._members = members
