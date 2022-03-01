from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.invalid_request_error import InvalidRequestError
from athenian.api.response import ResponseError


class CommonFilterProperties(Model, sealed=False):
    """Define `account`, `date_from`, `date_to`, and `timezone` properties."""

    openapi_types = {
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "account": int,
    }

    attribute_map = {
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "account": "account",
    }

    def __init__(
        self,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        account: Optional[int] = None,
    ):
        """CommonFilterProperties - a model defined in OpenAPI

        :param date_from: The date_from of this CommonFilterProperties.
        :param date_to: The date_to of this CommonFilterProperties.
        :param timezone: The timezone of this CommonFilterProperties.
        :param account: The account of this CommonFilterProperties.
        """
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._account = account

    @property
    def account(self) -> int:
        """Gets the account of this Model.

        :return: The account of this Model.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this Model.

        :param account: The account of this Model.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def date_from(self) -> date:
        """Gets the date_from of this Model.

        :return: The date_from of this Model.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date) -> None:
        """Sets the date_from of this Model.

        :param date_from: The date_from of this Model.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this Model.

        :return: The date_to of this Model.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date) -> None:
        """Sets the date_to of this Model.

        :param date_to: The date_to of this Model.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def timezone(self) -> int:
        """Gets the timezone of this Model.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :return: The timezone of this Model.
        """
        return self._timezone

    @timezone.setter
    def timezone(self, timezone: int) -> None:
        """Sets the timezone of this Model.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :param timezone: The timezone of this Model.
        """
        if timezone is not None and timezone > 720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value less than or equal to `720`")
        if timezone is not None and timezone < -720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value greater than or equal to `-720`")

        self._timezone = timezone

    def resolve_time_from_and_to(self) -> Tuple[datetime, datetime]:
        """Extract the time window from the request model: the timestamps of `from` and `to`."""
        if self.date_from > self.date_to:
            raise ResponseError(InvalidRequestError(
                "`date_to` may not be less than `date_from`",
                detail="from:%s > to:%s" % (self.date_from, self.date_to)))
        time_from = datetime.combine(self.date_from, datetime.min.time(), tzinfo=timezone.utc)
        time_to = datetime.combine(self.date_to, datetime.max.time(), tzinfo=timezone.utc)
        if self.timezone is not None:
            tzoffset = timedelta(minutes=-self.timezone)
            time_from += tzoffset
            time_to += tzoffset
        return time_from, time_to
