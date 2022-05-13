from typing import Optional

from athenian.api.models.web.base_model_ import Model


class WorkTypeGetRequest(Model):
    """Identifier of a work type - a set of rules to group PRs, releases, etc. together."""

    attribute_types = {"account": int, "name": str}

    attribute_map = {"account": "account", "name": "name"}

    def __init__(self, account: Optional[int] = None, name: Optional[str] = None):
        """WorkTypeGetRequest - a model defined in OpenAPI

        :param account: The account of this WorkTypeGetRequest.
        :param name: The name of this WorkTypeGetRequest.
        """
        self._account = account
        self._name = name

    @property
    def account(self) -> int:
        """Gets the account of this WorkTypeGetRequest.

        User's account ID.

        :return: The account of this WorkTypeGetRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this WorkTypeGetRequest.

        User's account ID.

        :param account: The account of this WorkTypeGetRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def name(self) -> str:
        """Gets the name of this WorkTypeGetRequest.

        Work type name. It is unique within the account scope.

        :return: The name of this WorkTypeGetRequest.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this WorkTypeGetRequest.

        Work type name. It is unique within the account scope.

        :param name: The name of this WorkTypeGetRequest.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")
        if len(name) < 1:
            raise ValueError(
                "Invalid value for `name`, length must be greater than or equal to `3`")

        self._name = name
