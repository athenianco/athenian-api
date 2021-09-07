from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class DeleteEventsCacheRequest(Model):
    """Definition of the cache reset operation."""

    openapi_types = {
        "account": int,
        "repositories": List[str],
        "targets": List[str],
    }

    attribute_map = {
        "account": "account",
        "repositories": "repositories",
        "targets": "targets",
    }

    def __init__(
        self,
        account: Optional[int] = None,
        repositories: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
    ):
        """DeleteEventsCacheRequest - a model defined in OpenAPI

        :param account: The account of this DeleteEventsCacheRequest.
        :param repositories: The repositories of this DeleteEventsCacheRequest.
        :param targets: The targets of this DeleteEventsCacheRequest.
        """
        self._account = account
        self._repositories = repositories
        self._targets = targets

    @property
    def account(self) -> int:
        """Gets the account of this DeleteEventsCacheRequest.

        Account ID.

        :return: The account of this DeleteEventsCacheRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this DeleteEventsCacheRequest.

        Account ID.

        :param account: The account of this DeleteEventsCacheRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def repositories(self) -> List[str]:
        """Gets the repositories of this DeleteEventsCacheRequest.

        Set of repositories. An empty list raises a bad response 400. Duplicates are automatically
        ignored.

        :return: The repositories of this DeleteEventsCacheRequest.
        """
        return self._repositories

    @repositories.setter
    def repositories(self, repositories: List[str]):
        """Sets the repositories of this DeleteEventsCacheRequest.

        Set of repositories. An empty list raises a bad response 400. Duplicates are automatically
        ignored.

        :param repositories: The repositories of this DeleteEventsCacheRequest.
        """
        if repositories is None:
            raise ValueError("Invalid value for `repositories`, must not be `None`")

        self._repositories = repositories

    @property
    def targets(self) -> List[str]:
        """Gets the targets of this DeleteEventsCacheRequest.

        Parts of the precomputed cache to reset.

        :return: The targets of this DeleteEventsCacheRequest.
        """
        return self._targets

    @targets.setter
    def targets(self, targets: List[str]):
        """Sets the targets of this DeleteEventsCacheRequest.

        Parts of the precomputed cache to reset.

        :param targets: The targets of this DeleteEventsCacheRequest.
        """
        allowed_values = {"release", "deployment"}
        if not set(targets).issubset(set(allowed_values)):
            raise ValueError(
                "Invalid values for `targets` [%s], must be a subset of [%s]" % (
                    ", ".join(set(targets) - set(allowed_values)),
                    ", ".join(allowed_values),
                ))

        self._targets = targets
