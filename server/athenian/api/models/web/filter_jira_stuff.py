from datetime import date
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class FilterJIRAStuff(Model):
    """Request of `/filter/jira` to retrieve epics and labels."""

    openapi_types = {
        "account": int,
        "date_from": date,
        "date_to": date,
        "timezone": int,
    }

    attribute_map = {
        "account": "account",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
    }

    def __init__(self,
                 account: Optional[int] = None,
                 date_from: Optional[date] = None,
                 date_to: Optional[date] = None,
                 timezone: Optional[int] = None):
        """FilterJIRAStuff - a model defined in OpenAPI

        :param account: The account of this FilterJIRAStuff.
        :param date_from: The date_from of this FilterJIRAStuff.
        :param date_to: The date_to of this FilterJIRAStuff.
        :param timezone: The timezone of this FilterJIRAStuff.
        """
        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone

    @property
    def account(self) -> int:
        """Gets the account of this FilterJIRAStuff.

        :return: The account of this FilterJIRAStuff.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this FilterJIRAStuff.

        :param account: The account of this FilterJIRAStuff.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def date_from(self) -> date:
        """Gets the date_from of this FilterJIRAStuff.

        :return: The date_from of this FilterJIRAStuff.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date):
        """Sets the date_from of this FilterJIRAStuff.

        :param date_from: The date_from of this FilterJIRAStuff.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this FilterJIRAStuff.

        :return: The date_to of this FilterJIRAStuff.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date):
        """Sets the date_to of this FilterJIRAStuff.

        :param date_to: The date_to of this FilterJIRAStuff.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def timezone(self) -> int:
        """Gets the timezone of this FilterJIRAStuff.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :return: The timezone of this FilterJIRAStuff.
        """
        return self._timezone

    @timezone.setter
    def timezone(self, timezone: int):
        """Sets the timezone of this FilterJIRAStuff.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :param timezone: The timezone of this FilterJIRAStuff.
        """
        if timezone is not None and timezone > 720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value less than or equal to `720`")
        if timezone is not None and timezone < -720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value greater than or equal to `-720`")

        self._timezone = timezone
