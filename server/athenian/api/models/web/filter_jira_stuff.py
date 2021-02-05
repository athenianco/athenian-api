from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.common_filter_properties import CommonFilterPropertiesMixin
from athenian.api.models.web.jira_filter_return import JIRAFilterReturn
from athenian.api.models.web.jira_filter_with import JIRAFilterWith


class FilterJIRAStuff(Model, CommonFilterPropertiesMixin):
    """Request of `/filter/jira` to retrieve epics and labels."""

    openapi_types = {
        "account": int,
        "date_from": Optional[date],
        "date_to": Optional[date],
        "timezone": int,
        "priorities": Optional[List[str]],
        "labels_include": Optional[List[str]],
        "labels_exclude": Optional[List[str]],
        "with_": Optional[JIRAFilterWith],
        "exclude_inactive": bool,
        "return_": Optional[List[str]],
    }

    attribute_map = {
        "account": "account",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "priorities": "priorities",
        "labels_include": "labels_include",
        "labels_exclude": "labels_exclude",
        "with_": "with",
        "exclude_inactive": "exclude_inactive",
        "return_": "return",
    }

    def __init__(self,
                 account: Optional[int] = None,
                 date_from: Optional[date] = None,
                 date_to: Optional[date] = None,
                 timezone: Optional[int] = None,
                 priorities: Optional[List[str]] = None,
                 labels_include: Optional[List[str]] = None,
                 labels_exclude: Optional[List[str]] = None,
                 with_: Optional[JIRAFilterWith] = None,
                 exclude_inactive: Optional[bool] = None,
                 return_: Optional[List[str]] = None,
                 ):
        """FilterJIRAStuff - a model defined in OpenAPI

        :param account: The account of this FilterJIRAStuff.
        :param date_from: The date_from of this FilterJIRAStuff.
        :param date_to: The date_to of this FilterJIRAStuff.
        :param timezone: The timezone of this FilterJIRAStuff.
        :param priorities: The priorities of this FilterJIRAStuff.
        :param labels_include: The labels_include of this FilterJIRAStuff.
        :param labels_exclude: The labels_exclude of this FilterJIRAStuff.
        :param with_: The with_ of this FilterJIRAStuff.
        :param exclude_inactive: The exclude_inactive of this FilterJIRAStuff.
        :param return_: The return of this FilterJIRAStuff.
        """
        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._priorities = priorities
        self._labels_include = labels_include
        self._labels_exclude = labels_exclude
        self._with_ = with_
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
        if diff := set(return_ or []) - set(JIRAFilterReturn):
            raise ValueError("`return` contains invalid values: %s" % diff)
        self._return_ = return_

    @property
    def priorities(self) -> Optional[List[str]]:
        """Gets the priorities of this FilterJIRAStuff.

        Selected issue priorities.

        :return: The priorities of this FilterJIRAStuff.
        """
        return self._priorities

    @priorities.setter
    def priorities(self, priorities: Optional[List[str]]):
        """Sets the priorities of this FilterJIRAStuff.

        Selected issue priorities.

        :param priorities: The priorities of this FilterJIRAStuff.
        """
        self._priorities = priorities

    @property
    def with_(self) -> Optional[JIRAFilterWith]:
        """Gets the with of this FilterJIRAStuff.

        JIRA issue participants.

        :return: The with of this FilterJIRAStuff.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[JIRAFilterWith]):
        """Sets the with of this FilterJIRAStuff.

        JIRA issue participants.

        :param with_: The with of this FilterJIRAStuff.
        """
        self._with_ = with_

    @property
    def labels_include(self) -> Optional[List[str]]:
        """Gets the labels_include of this FilterJIRAStuff.

        JIRA issues must contain at least one label from the list.
        Several labels may be concatenated by a comma `,` so that all of them
        are required.

        :return: The labels_include of this FilterJIRAStuff.
        """
        return self._labels_include

    @labels_include.setter
    def labels_include(self, labels_include: Optional[List[str]]):
        """Sets the labels_include of this FilterJIRAStuff.

        JIRA issues must contain at least one label from the list.
        Several labels may be concatenated by a comma `,` so that all of them
        are required.

        :param labels_include: The labels_include of this FilterJIRAStuff.
        """
        self._labels_include = labels_include

    @property
    def labels_exclude(self) -> Optional[List[str]]:
        """Gets the labels_exclude of this FilterJIRAStuff.

        JIRA issues may not contain labels from this list.

        :return: The labels_exclude of this FilterJIRAStuff.
        """
        return self._labels_exclude

    @labels_exclude.setter
    def labels_exclude(self, labels_exclude: Optional[List[str]]):
        """Sets the labels_exclude of this FilterJIRAStuff.

        JIRA issues may not contain labels from this list.

        :param labels_exclude: The labels_exclude of this FilterJIRAStuff.
        """
        self._labels_exclude = labels_exclude
