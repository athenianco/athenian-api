from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.common_filter_properties import CommonFilterPropertiesMixin


class FilterJIRACommon(Model, CommonFilterPropertiesMixin):
    """Common properies if a JIRA issue or epic."""

    openapi_types = {
        "account": int,
        "timezone": int,
        "priorities": Optional[List[str]],
        "types": Optional[List[str]],
        "labels_include": Optional[List[str]],
        "labels_exclude": Optional[List[str]],
        "exclude_inactive": bool,
    }

    attribute_map = {
        "account": "account",
        "timezone": "timezone",
        "priorities": "priorities",
        "types": "types",
        "labels_include": "labels_include",
        "labels_exclude": "labels_exclude",
        "exclude_inactive": "exclude_inactive",
    }

    __enable_slots__ = False

    def __init__(
        self,
        account: Optional[int] = None,
        timezone: Optional[int] = None,
        priorities: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        labels_include: Optional[List[str]] = None,
        labels_exclude: Optional[List[str]] = None,
        exclude_inactive: Optional[bool] = None,
    ):
        """FilterJIRACommon - a model defined in OpenAPI

        :param account: The account of this FilterJIRACommon.
        :param timezone: The timezone of this FilterJIRACommon.
        :param priorities: The priorities of this FilterJIRACommon.
        :param types: The types of this FilterJIRACommon.
        :param labels_include: The labels_include of this FilterJIRACommon.
        :param labels_exclude: The labels_exclude of this FilterJIRACommon.
        :param exclude_inactive: The exclude_inactive of this FilterJIRACommon.
        """
        self._account = account
        self._timezone = timezone
        self._priorities = priorities
        self._types = types
        self._labels_include = labels_include
        self._labels_exclude = labels_exclude
        self._exclude_inactive = exclude_inactive

    @property
    def priorities(self) -> Optional[List[str]]:
        """Gets the priorities of this FilterJIRACommon.

        Selected issue priorities.

        :return: The priorities of this FilterJIRACommon.
        """
        return self._priorities

    @priorities.setter
    def priorities(self, priorities: Optional[List[str]]):
        """Sets the priorities of this FilterJIRACommon.

        Selected issue priorities.

        :param priorities: The priorities of this FilterJIRACommon.
        """
        self._priorities = priorities

    @property
    def types(self) -> Optional[List[str]]:
        """Gets the types of this FilterJIRACommon.

        Selected issue types. Ignored for epics.

        :return: The types of this FilterJIRACommon.
        """
        return self._types

    @types.setter
    def types(self, types: Optional[List[str]]):
        """Sets the types of this FilterJIRACommon.

        Selected issue types. Ignored for epics.

        :param types: The types of this FilterJIRACommon.
        """
        self._types = types

    @property
    def labels_include(self) -> Optional[List[str]]:
        """Gets the labels_include of this FilterJIRACommon.

        JIRA issues must contain at least one label from the list. Several labels may be
        concatenated by a comma `,` so that all of them are required.

        :return: The labels_include of this FilterJIRACommon.
        """
        return self._labels_include

    @labels_include.setter
    def labels_include(self, labels_include: Optional[List[str]]):
        """Sets the labels_include of this FilterJIRACommon.

        JIRA issues must contain at least one label from the list. Several labels may be
        concatenated by a comma `,` so that all of them are required.

        :param labels_include: The labels_include of this FilterJIRACommon.
        """
        self._labels_include = labels_include

    @property
    def labels_exclude(self) -> Optional[List[str]]:
        """Gets the labels_exclude of this FilterJIRACommon.

        JIRA issues may not contain labels from this list.

        :return: The labels_exclude of this FilterJIRACommon.
        """
        return self._labels_exclude

    @labels_exclude.setter
    def labels_exclude(self, labels_exclude: Optional[List[str]]):
        """Sets the labels_exclude of this FilterJIRACommon.

        JIRA issues may not contain labels from this list.

        :param labels_exclude: The labels_exclude of this FilterJIRACommon.
        """
        self._labels_exclude = labels_exclude

    @property
    def exclude_inactive(self) -> bool:
        """Gets the exclude_inactive of this FilterJIRACommon.

        Value indicating whether issues with the last update older than `date_from` should be
        ignored. If `date_from` and `date_to` are `null`, does nothing.

        :return: The exclude_inactive of this FilterJIRACommon.
        """
        return self._exclude_inactive

    @exclude_inactive.setter
    def exclude_inactive(self, exclude_inactive: bool):
        """Sets the exclude_inactive of this FilterJIRACommon.

        Value indicating whether issues with the last update older than `date_from` should be
        ignored. If `date_from` and `date_to` are `null`, does nothing.

        :param exclude_inactive: The exclude_inactive of this FilterJIRACommon.
        """
        if exclude_inactive is None:
            raise ValueError("Invalid value for `exclude_inactive`, must not be `None`")

        self._exclude_inactive = exclude_inactive
