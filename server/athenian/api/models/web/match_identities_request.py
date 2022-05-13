from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.contributor_identity import ContributorIdentity


class MatchIdentitiesRequest(Model):
    """Request body of `/match/identities`."""

    attribute_types = {"account": int, "identities": List[ContributorIdentity]}

    attribute_map = {"account": "account", "identities": "identities"}

    def __init__(self,
                 account: Optional[int] = None,
                 identities: Optional[List[ContributorIdentity]] = None):
        """MatchIdentitiesRequest - a model defined in OpenAPI

        :param account: The account of this MatchIdentitiesRequest.
        :param identities: The identities of this MatchIdentitiesRequest.
        """
        self._account = account
        self._identities = identities

    @property
    def account(self) -> int:
        """Gets the account of this MatchIdentitiesRequest.

        User's account ID.

        :return: The account of this MatchIdentitiesRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this MatchIdentitiesRequest.

        User's account ID.

        :param account: The account of this MatchIdentitiesRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def identities(self) -> List[ContributorIdentity]:
        """Gets the identities of this MatchIdentitiesRequest.

        :return: The identities of this MatchIdentitiesRequest.
        """
        return self._identities

    @identities.setter
    def identities(self, identities: List[ContributorIdentity]):
        """Sets the identities of this MatchIdentitiesRequest.

        :param identities: The identities of this MatchIdentitiesRequest.
        """
        if identities is None:
            raise ValueError("Invalid value for `identities`, must not be `None`")

        self._identities = identities
