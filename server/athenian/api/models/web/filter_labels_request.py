from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class FilterLabelsRequest(Model):
    """
    Request body of `/filter/labels`.

    Defines the account and the repositories where to look for the labels.
    """

    attribute_types = {"account": int, "repositories": List[str]}
    attribute_map = {"account": "account", "repositories": "repositories"}

    def __init__(self, account: Optional[int] = None, repositories: Optional[List[str]] = None):
        """FilterLabelsRequest - a model defined in OpenAPI

        :param account: The account of this FilterLabelsRequest.
        :param repositories: The repositories of this FilterLabelsRequest.
        """
        self._account = account
        self._repositories = repositories

    @property
    def account(self) -> int:
        """Gets the account of this FilterLabelsRequest.

        :return: The account of this FilterLabelsRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this FilterLabelsRequest.

        :param account: The account of this FilterLabelsRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def repositories(self) -> List[str]:
        """Gets the repositories of this FilterLabelsRequest.

        :return: The repositories of this FilterLabelsRequest.
        """
        return self._repositories

    @repositories.setter
    def repositories(self, repositories: List[str]):
        """Sets the repositories of this FilterLabelsRequest.

        :param repositories: The repositories of this FilterLabelsRequest.
        """
        self._repositories = repositories
