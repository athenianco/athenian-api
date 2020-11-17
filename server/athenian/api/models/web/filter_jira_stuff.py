from datetime import date
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.common_filter_properties import CommonFilterPropertiesMixin


class FilterJIRAStuff(Model, CommonFilterPropertiesMixin):
    """Request of `/filter/jira` to retrieve epics and labels."""

    openapi_types = {
        "account": int,
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "exclude_inactive": bool,
    }

    attribute_map = {
        "account": "account",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "exclude_inactive": "exclude_inactive",
    }

    def __init__(self,
                 account: Optional[int] = None,
                 date_from: Optional[date] = None,
                 date_to: Optional[date] = None,
                 timezone: Optional[int] = None,
                 exclude_inactive: Optional[bool] = None):
        """FilterJIRAStuff - a model defined in OpenAPI

        :param account: The account of this FilterJIRAStuff.
        :param date_from: The date_from of this FilterJIRAStuff.
        :param date_to: The date_to of this FilterJIRAStuff.
        :param timezone: The timezone of this FilterJIRAStuff.
        :param exclude_inactive: The exclude_inactive of this FilterJIRAStuff.
        """
        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._exclude_inactive = exclude_inactive

    @property
    def exclude_inactive(self) -> bool:
        """Gets the exclude_inactive of this FilterJIRAStuff.

        Value indicating whether issues with the last update older than `date_from` should be \
        ignored.

        :return: The exclude_inactive of this FilterJIRAStuff.
        """
        return self._exclude_inactive

    @exclude_inactive.setter
    def exclude_inactive(self, exclude_inactive: bool):
        """Sets the exclude_inactive of this FilterJIRAStuff.

        Value indicating whether issues with the last update older than `date_from` should be \
        ignored.

        :param exclude_inactive: The exclude_inactive of this FilterJIRAStuff.
        """
        if exclude_inactive is None:
            raise ValueError("Invalid value for `exclude_inactive`, must not be `None`")

        self._exclude_inactive = exclude_inactive
