from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.common_filter_properties import CommonFilterPropertiesMixin
from athenian.api.models.web.found_jira_stuff import FoundJIRAStuff


class FilterJIRAStuff(Model, CommonFilterPropertiesMixin):
    """Request of `/filter/jira` to retrieve epics and labels."""

    openapi_types = {
        "account": int,
        "date_from": Optional[date],
        "date_to": Optional[date],
        "timezone": int,
        "exclude_inactive": bool,
        "return_": Optional[List[str]],
    }

    attribute_map = {
        "account": "account",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "exclude_inactive": "exclude_inactive",
        "return_": "return",
    }

    def __init__(self,
                 account: Optional[int] = None,
                 date_from: Optional[date] = None,
                 date_to: Optional[date] = None,
                 timezone: Optional[int] = None,
                 exclude_inactive: Optional[bool] = None,
                 return_: Optional[List[str]] = None,
                 ):
        """FilterJIRAStuff - a model defined in OpenAPI

        :param account: The account of this FilterJIRAStuff.
        :param date_from: The date_from of this FilterJIRAStuff.
        :param date_to: The date_to of this FilterJIRAStuff.
        :param timezone: The timezone of this FilterJIRAStuff.
        :param exclude_inactive: The exclude_inactive of this FilterJIRAStuff.
        :param return_: The return of this FilterJIRAStuff.
        """
        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._exclude_inactive = exclude_inactive
        self._return_ = return_

    # We have to redefine these to assign Optional.

    @property
    def date_from(self) -> Optional[date]:
        """Gets the date_from of this Model.

        `null` disables the time filter boundary, thus everything is returned. `date_from` and
        `date_to` must be both either `null` or not `null`.

        :return: The date_from of this Model.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: Optional[date]) -> None:
        """Sets the date_from of this Model.

        `null` disables the time filter boundary, thus everything is returned. `date_from` and
        `date_to` must be both either `null` or not `null`.

        :param date_from: The date_from of this Model.
        """
        self._date_from = date_from

    @property
    def date_to(self) -> Optional[date]:
        """Gets the date_to of this Model.

        `null` disables the time filter boundary, thus everything is returned. `date_from` and
        `date_to` must be both either `null` or not `null`.

        :return: The date_to of this Model.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: Optional[date]) -> None:
        """Sets the date_to of this Model.

        `null` disables the time filter boundary, thus everything is returned. `date_from` and
        `date_to` must be both either `null` or not `null`.

        :param date_to: The date_to of this Model.
        """
        self._date_to = date_to

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

    @property
    def return_(self) -> Optional[List[str]]:
        """Gets the return of this FilterJIRAStuff.

        Specifies which items are required, an empty or missing array means everything.

        :return: The return of this FilterJIRAStuff.
        """
        return self._return_

    @return_.setter
    def return_(self, return_: Optional[List[str]]) -> None:
        """Sets the return of this FilterJIRAStuff.

        Specifies which items are required, an empty or missing array means everything.

        :param return_: The return of this FilterJIRAStuff.
        """
        if diff := set(return_ or []) - FoundJIRAStuff.openapi_types.keys():
            raise ValueError("`return` contains invalid values: %s" % diff)
        self._return_ = return_
