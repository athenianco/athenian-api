from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class TeamCreateRequest(Model):
    """Team creation request."""

    attribute_types = {
        "account": int,
        "name": str,
        "members": List[str],
        "parent": int,
    }

    attribute_map = {
        "account": "account",
        "name": "name",
        "members": "members",
        "parent": "parent",
    }

    def __init__(
        self,
        account: Optional[int] = None,
        name: Optional[str] = None,
        members: Optional[List[str]] = None,
        parent: Optional[int] = None,
    ):
        """TeamCreateRequest - a model defined in OpenAPI

        :param account: The account of this TeamCreateRequest.
        :param name: The name of this TeamCreateRequest.
        :param members: The members of this TeamCreateRequest.
        :param parent: The parent of this TeamCreateRequest.
        """
        self._account = account
        self._name = name
        self._members = members
        self._parent = parent

    @property
    def account(self) -> int:
        """Gets the account of this TeamCreateRequest.

        Account identifier. That account will own the created team.

        :return: The account of this TeamCreateRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this TeamCreateRequest.

        Account identifier. That account will own the created team.

        :param account: The account of this TeamCreateRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def name(self) -> str:
        """Gets the name of this TeamCreateRequest.

        Name of the team.

        :return: The name of this TeamCreateRequest.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this TeamCreateRequest.

        Name of the team.

        :param name: The name of this TeamCreateRequest.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def members(self) -> List[str]:
        """Gets the members of this TeamCreateRequest.

        A set of developers. An empty list disables the filter and includes everybody.
        Duplicates are automatically ignored.

        :return: The members of this TeamCreateRequest.
        """
        return self._members

    @members.setter
    def members(self, members: List[str]):
        """Sets the members of this TeamCreateRequest.

        A set of developers. An empty list disables the filter and includes everybody.
        Duplicates are automatically ignored.

        :param members: The members of this TeamCreateRequest.
        """
        if members is None:
            raise ValueError("Invalid value for `members`, must not be `None`")

        self._members = members

    @property
    def parent(self) -> Optional[int]:
        """Gets the parent of this TeamCreateRequest.

        Identifier of the higher level team.

        :return: The parent of this TeamCreateRequest.
        """
        return self._parent

    @parent.setter
    def parent(self, parent: Optional[int]):
        """Sets the parent of this TeamCreateRequest.

        Identifier of the higher level team.

        :param parent: The parent of this TeamCreateRequest.
        """
        self._parent = parent
