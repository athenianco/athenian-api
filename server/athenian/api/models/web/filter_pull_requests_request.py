from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.common_filter_properties import CommonFilterPropertiesMixin
from athenian.api.models.web.jira_filter import JIRAFilter
from athenian.api.models.web.pull_request_event import PullRequestEvent
from athenian.api.models.web.pull_request_property import PullRequestProperty
from athenian.api.models.web.pull_request_stage import PullRequestStage
from athenian.api.models.web.pull_request_with import PullRequestWith


class FilterPullRequestsRequest(Model, CommonFilterPropertiesMixin):
    """PR filters for /filter/pull_requests."""

    openapi_types = {
        "account": int,
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "in_": List[str],
        "properties": List[str],
        "events": List[str],
        "stages": List[str],
        "with_": PullRequestWith,
        "labels_include": List[str],
        "labels_exclude": List[str],
        "exclude_inactive": bool,
        "jira": JIRAFilter,
        "limit": int,
    }

    attribute_map = {
        "account": "account",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "in_": "in",
        "properties": "properties",
        "events": "events",
        "stages": "stages",
        "with_": "with",
        "labels_include": "labels_include",
        "labels_exclude": "labels_exclude",
        "exclude_inactive": "exclude_inactive",
        "jira": "jira",
        "limit": "limit",
    }

    def __init__(
        self,
        account: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        in_: Optional[List[str]] = None,
        properties: Optional[List[str]] = None,
        events: Optional[List[str]] = None,
        stages: Optional[List[str]] = None,
        with_: Optional[PullRequestWith] = None,
        labels_include: Optional[List[str]] = None,
        labels_exclude: Optional[List[str]] = None,
        exclude_inactive: Optional[bool] = None,
        jira: Optional[JIRAFilter] = None,
        limit: Optional[int] = None,
    ):
        """FilterPullRequestsRequest - a model defined in OpenAPI

        :param account: The account of this FilterPullRequestsRequest.
        :param date_from: The date_from of this FilterPullRequestsRequest.
        :param date_to: The date_to of this FilterPullRequestsRequest.
        :param timezone: The timezone of this FilterPullRequestsRequest.
        :param in_: The in_ of this FilterPullRequestsRequest.
        :param properties: The properties of this FilterPullRequestsRequest.
        :param events: The events of this FilterPullRequestsRequest.
        :param stages: The stages of this FilterPullRequestsRequest.
        :param with_: The with_ of this FilterPullRequestsRequest.
        :param labels_include: The labels_include of this FilterPullRequestsRequest.
        :param labels_exclude: The labels_exclude of this FilterPullRequestsRequest.
        :param exclude_inactive: The exclude_inactive of this FilterPullRequestsRequest.
        :param jira: The jira of this FilterPullRequestsRequest.
        :param limit: The limit of this FilterPullRequestsRequest.
        """
        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._in_ = in_
        self._properties = properties
        self._events = events
        self._stages = stages
        self._with_ = with_
        self._labels_include = labels_include
        self._labels_exclude = labels_exclude
        self._exclude_inactive = exclude_inactive
        self._jira = jira
        self._limit = limit

    @property
    def in_(self) -> List[str]:
        """Gets the in_ of this FilterPullRequestsRequest.

        :return: The in_ of this FilterPullRequestsRequest.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in of this FilterPullRequestsRequest.

        :param in_: The in of this FilterPullRequestsRequest.
        """
        if in_ is None:
            raise ValueError("Invalid value for `in`, must not be `None`")

        self._in_ = in_

    @property
    def properties(self) -> List[str]:
        """Gets the properties of this FilterPullRequestsRequest.

        :return: The properties of this FilterPullRequestsRequest.
        """
        return self._properties

    @properties.setter
    def properties(self, properties: List[str]):
        """Sets the properties of this FilterPullRequestsRequest.

        :param properties: The properties of this FilterPullRequestsRequest.
        """
        if properties is None:
            raise ValueError("Invalid value for `properties`, must not be `None`")

        for stage in properties:
            if stage not in PullRequestProperty:
                raise ValueError("Invalid property: %s" % stage)

        self._properties = properties

    @property
    def events(self) -> List[str]:
        """Gets the events of this FilterPullRequestsRequest.

        :return: The events of this FilterPullRequestsRequest.
        """
        return self._events

    @events.setter
    def events(self, events: List[str]):
        """Sets the events of this FilterPullRequestsRequest.

        :param events: The events of this FilterPullRequestsRequest.
        """
        if events is None:
            raise ValueError("Invalid value for `events`, must not be `None`")

        for stage in events:
            if stage not in PullRequestEvent:
                raise ValueError("Invalid stage: %s" % stage)

        self._events = events

    @property
    def stages(self) -> List[str]:
        """Gets the stages of this FilterPullRequestsRequest.

        :return: The stages of this FilterPullRequestsRequest.
        """
        return self._stages

    @stages.setter
    def stages(self, stages: List[str]):
        """Sets the stages of this FilterPullRequestsRequest.

        :param stages: The stages of this FilterPullRequestsRequest.
        """
        if stages is None:
            raise ValueError("Invalid value for `stages`, must not be `None`")

        for stage in stages:
            if stage not in PullRequestStage:
                raise ValueError("Invalid stage: %s" % stage)

        self._stages = stages

    @property
    def with_(self) -> Optional[PullRequestWith]:
        """Gets the with_ of this FilterPullRequestsRequest.

        :return: The with_ of this FilterPullRequestsRequest.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: PullRequestWith):
        """Sets the with_ of this FilterPullRequestsRequest.

        :param with_: The with_ of this FilterPullRequestsRequest.
        """
        self._with_ = with_

    @property
    def labels_include(self) -> Optional[List[str]]:
        """Gets the labels_include of this FilterPullRequestsRequest.

        :return: The labels_include of this FilterPullRequestsRequest.
        """
        return self._labels_include

    @labels_include.setter
    def labels_include(self, labels_include: List[str]):
        """Sets the labels_include of this FilterPullRequestsRequest.

        :param labels_include: The labels_include of this FilterPullRequestsRequest.
        """
        self._labels_include = labels_include

    @property
    def labels_exclude(self) -> Optional[List[str]]:
        """Gets the labels_exclude of this FilterPullRequestsRequest.

        :return: The labels_exclude of this FilterPullRequestsRequest.
        """
        return self._labels_exclude

    @labels_exclude.setter
    def labels_exclude(self, labels_exclude: List[str]):
        """Sets the labels_exclude of this FilterPullRequestsRequest.

        :param labels_exclude: The labels_exclude of this FilterPullRequestsRequest.
        """
        self._labels_exclude = labels_exclude

    @property
    def exclude_inactive(self) -> bool:
        """Gets the exclude_inactive of this FilterPullRequestsRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :return: The exclude_inactive of this FilterPullRequestsRequest.
        """
        return self._exclude_inactive

    @exclude_inactive.setter
    def exclude_inactive(self, exclude_inactive: bool):
        """Sets the exclude_inactive of this FilterPullRequestsRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :param exclude_inactive: The exclude_inactive of this FilterPullRequestsRequest.
        """
        if exclude_inactive is None:
            raise ValueError("Invalid value for `exclude_inactive`, must not be `None`")

        self._exclude_inactive = exclude_inactive

    @property
    def jira(self) -> Optional[JIRAFilter]:
        """Gets the jira of this FilterPullRequestsRequest.

        :return: The jira of this FilterPullRequestsRequest.
        """
        return self._jira

    @jira.setter
    def jira(self, jira: Optional[JIRAFilter]):
        """Sets the jira of this FilterPullRequestsRequest.

        :param jira: The jira of this FilterPullRequestsRequest.
        """
        self._jira = jira

    @property
    def limit(self) -> int:
        """Gets the limit of this FilterPullRequestsRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :return: The limit of this FilterPullRequestsRequest.
        """
        return self._limit

    @limit.setter
    def limit(self, limit: int):
        """Sets the limit of this FilterPullRequestsRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :param limit: The limit of this FilterPullRequestsRequest.
        """
        if limit is not None and limit < 1:
            raise ValueError("`limit` must be greater than 0: %s" % limit)

        self._limit = limit
