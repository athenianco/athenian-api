from typing import Dict, List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.release_pair import ReleasePair


class DiffReleasesRequest(Model):
    """Request of `/diff/releases`. Define pairs of releases for several repositories to find \
    the releases in between."""

    attribute_types = {"account": int, "borders": Dict[str, List[ReleasePair]]}
    attribute_map = {"account": "account", "borders": "borders"}

    def __init__(
        self,
        account: Optional[int] = None,
        borders: Optional[Dict[str, List[ReleasePair]]] = None,
    ):
        """DiffReleasesRequest - a model defined in OpenAPI

        :param account: The account of this DiffReleasesRequest.
        :param borders: The borders of this DiffReleasesRequest.
        """
        self._account = account
        self._borders = borders

    @property
    def account(self) -> int:
        """Gets the account of this DiffReleasesRequest.

        :return: The account of this DiffReleasesRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this DiffReleasesRequest.

        :param account: The account of this DiffReleasesRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def borders(self) -> Dict[str, List[ReleasePair]]:
        """Gets the borders of this DiffReleasesRequest.

        Mapping from repository names to analyzed release pairs.

        :return: The borders of this DiffReleasesRequest.
        """
        return self._borders

    @borders.setter
    def borders(self, borders: Dict[str, List[ReleasePair]]):
        """Sets the borders of this DiffReleasesRequest.

        Mapping from repository names to analyzed release pairs.

        :param borders: The borders of this DiffReleasesRequest.
        """
        if borders is None:
            raise ValueError("Invalid value for `borders`, must not be `None`")

        self._borders = borders
