from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.release_names import ReleaseNames


class GetReleasesRequest(Model):
    """Request body of `/get/releases`. Declaration of which releases the user wants to list."""

    attribute_types = {"account": int, "releases": List[ReleaseNames]}
    attribute_map = {"account": "account", "releases": "releases"}

    def __init__(
        self,
        account: Optional[int] = None,
        releases: Optional[List[ReleaseNames]] = None,
    ):
        """GetReleasesRequest - a model defined in OpenAPI

        :param account: The account of this GetReleasesRequest.
        :param releases: The releases of this GetReleasesRequest.
        """
        self._account = account
        self._releases = releases

    @property
    def account(self) -> int:
        """Gets the account of this GetReleasesRequest.

        Account ID.

        :return: The account of this GetReleasesRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this GetReleasesRequest.

        Account ID.

        :param account: The account of this GetReleasesRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def releases(self) -> List[ReleaseNames]:
        """Gets the releases of this GetReleasesRequest.

        List of repositories and release names to list.

        :return: The releases of this GetReleasesRequest.
        """
        return self._releases

    @releases.setter
    def releases(self, releases: List[ReleaseNames]):
        """Sets the releases of this GetReleasesRequest.

        List of repositories and release names to list.

        :param releases: The releases of this GetReleasesRequest.
        """
        if releases is None:
            raise ValueError("Invalid value for `releases`, must not be `None`")

        self._releases = releases
