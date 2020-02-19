from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.user import User


class Account(Model):
    """Account members: admins and regular users."""

    def __init__(self, admins: Optional[List[User]] = None, regulars: Optional[List[User]] = None):
        """Account - a model defined in OpenAPI

        :param admins: The admins of this Account.
        :param regulars: The regulars of this Account.
        """
        self.openapi_types = {"admins": List[User], "regulars": List[User]}

        self.attribute_map = {"admins": "admins", "regulars": "regulars"}

        self._admins = admins
        self._regulars = regulars

    @property
    def admins(self) -> List[User]:
        """Gets the admins of this Account.

        :return: The admins of this Account.
        """
        return self._admins

    @admins.setter
    def admins(self, admins: List[User]):
        """Sets the admins of this Account.

        :param admins: The admins of this Account.
        """
        if admins is None:
            raise ValueError("Invalid value for `admins`, must not be `None`")

        self._admins = admins

    @property
    def regulars(self) -> List[User]:
        """Gets the regulars of this Account.

        :return: The regulars of this Account.
        :rtype: List[User]
        """
        return self._regulars

    @regulars.setter
    def regulars(self, regulars: List[User]):
        """Sets the regulars of this Account.

        :param regulars: The regulars of this Account.
        """
        if regulars is None:
            raise ValueError("Invalid value for `regulars`, must not be `None`")

        self._regulars = regulars
