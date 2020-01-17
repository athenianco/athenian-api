from typing import List, Optional

from athenian.api import serialization
from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.user import User


class Team(Model):
    """Team members: admins and regular users."""

    def __init__(self, admins: Optional[List[User]] = None, regulars: Optional[List[User]] = None):
        """Team - a model defined in OpenAPI

        :param admins: The admins of this Team.
        :param regulars: The regulars of this Team.
        """
        self.openapi_types = {"admins": List[User], "regulars": List[User]}

        self.attribute_map = {"admins": "admins", "regulars": "regulars"}

        self._admins = admins
        self._regulars = regulars

    @classmethod
    def from_dict(cls, dikt: dict) -> "Team":
        """Returns the dict as a model

        :param dikt: A dict.
        :return: The Team of this Team.
        """
        return serialization.deserialize_model(dikt, cls)

    @property
    def admins(self) -> List[User]:
        """Gets the admins of this Team.

        :return: The admins of this Team.
        """
        return self._admins

    @admins.setter
    def admins(self, admins: List[User]):
        """Sets the admins of this Team.

        :param admins: The admins of this Team.
        """
        if admins is None:
            raise ValueError("Invalid value for `admins`, must not be `None`")

        self._admins = admins

    @property
    def regulars(self) -> List[User]:
        """Gets the regulars of this Team.

        :return: The regulars of this Team.
        :rtype: List[User]
        """
        return self._regulars

    @regulars.setter
    def regulars(self, regulars: List[User]):
        """Sets the regulars of this Team.

        :param regulars: The regulars of this Team.
        """
        if regulars is None:
            raise ValueError("Invalid value for `regulars`, must not be `None`")

        self._regulars = regulars
