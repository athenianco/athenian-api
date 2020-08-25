from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class FilterRepositoriesRequest(Model):
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

    __slots__ = ["_" + k for k in openapi_types]

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
    def account(self) -> int:
        """Gets the account of this FilterRepositoriesRequest.

        Session account ID.

        :return: The account of this FilterRepositoriesRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this FilterRepositoriesRequest.

        Session account ID.

        :param account: The account of this FilterRepositoriesRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def date_from(self) -> date:
        """Gets the date_from of this FilterRepositoriesRequest.

        Updates must be later than or equal to this date.

        :return: The date_from of this FilterRepositoriesRequest.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date):
        """Sets the date_from of this FilterRepositoriesRequest.

        Updates must be later than or equal to this date.

        :param date_from: The date_from of this FilterRepositoriesRequest.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this FilterRepositoriesRequest.

        Updates must be earlier than or equal to this date.

        :return: The date_to of this FilterRepositoriesRequest.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date):
        """Sets the date_to of this FilterRepositoriesRequest.

        Updates must be earlier than or equal to this date.

        :param date_to: The date_to of this FilterRepositoriesRequest.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def timezone(self) -> int:
        """Gets the timezone of this FilterRepositoriesRequest.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :return: The timezone of this FilterRepositoriesRequest.
        """
        return self._timezone

    @timezone.setter
    def timezone(self, timezone: int):
        """Sets the timezone of this FilterRepositoriesRequest.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :param timezone: The timezone of this FilterRepositoriesRequest.
        """
        if timezone is not None and timezone > 720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value less than or equal to `720`")
        if timezone is not None and timezone < -720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value greater than or equal to `-720`")

        self._timezone = timezone

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
