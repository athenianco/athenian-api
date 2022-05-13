from typing import Optional

from athenian.api.models.web.base_model_ import Model


class CreateTokenRequest(Model):
    """Request body of `/token/create` - creating a new Personal Access Token."""

    attribute_types = {"account": int, "name": str}
    attribute_map = {"account": "account", "name": "name"}

    def __init__(self, account: Optional[int] = None, name: Optional[str] = None):
        """CreateTokenRequest - a model defined in OpenAPI

        :param account: The account of this CreateTokenRequest.
        :param name: The name of this CreateTokenRequest.
        """
        self._account = account
        self._name = name

    @property
    def account(self) -> int:
        """Gets the account of this CreateTokenRequest.

        :return: The account of this CreateTokenRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this CreateTokenRequest.

        :param account: The account of this CreateTokenRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def name(self) -> str:
        """Gets the name of this CreateTokenRequest.

        :return: The name of this CreateTokenRequest.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this CreateTokenRequest.

        :param name: The name of this CreateTokenRequest.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name
