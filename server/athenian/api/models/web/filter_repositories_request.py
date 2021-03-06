from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.common_filter_properties import CommonFilterPropertiesMixin


class FilterRepositoriesRequest(Model, CommonFilterPropertiesMixin):
    """Structure to specify the filter traits of repositories."""

    openapi_types = {
        "account": int,
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "in_": List[str],
        "exclude_inactive": bool,
    }

    attribute_map = {
        "account": "account",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "in_": "in",
        "exclude_inactive": "exclude_inactive",
    }

    def __init__(
        self,
        account: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        in_: Optional[List[str]] = None,
        exclude_inactive: Optional[bool] = None,
    ):
        """FilterRepositoriesRequest - a model defined in OpenAPI

        :param account: The account of this FilterRepositoriesRequest.
        :param date_from: The date_from of this FilterRepositoriesRequest.
        :param date_to: The date_to of this FilterRepositoriesRequest.
        :param timezone: The timezone of this FilterRepositoriesRequest.
        :param in_: The in of this FilterRepositoriesRequest.
        :param exclude_inactive: The exclude_inactive of this FilterRepositoriesRequest.
        """
        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._in_ = in_
        self._exclude_inactive = exclude_inactive

    @property
    def in_(self) -> List[str]:
        """Gets the in_ of this FilterRepositoriesRequest.

        :return: The in_ of this FilterRepositoriesRequest.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in_ of this FilterRepositoriesRequest.

        :param in_: The in_ of this FilterRepositoriesRequest.
        """
        self._in_ = in_

    @property
    def exclude_inactive(self) -> bool:
        """Gets the exclude_inactive of this FilterRepositoriesRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :return: The exclude_inactive of this FilterRepositoriesRequest.
        """
        return self._exclude_inactive

    @exclude_inactive.setter
    def exclude_inactive(self, exclude_inactive: bool):
        """Sets the exclude_inactive of this FilterRepositoriesRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :param exclude_inactive: The exclude_inactive of this FilterRepositoriesRequest.
        """
        self._exclude_inactive = exclude_inactive
