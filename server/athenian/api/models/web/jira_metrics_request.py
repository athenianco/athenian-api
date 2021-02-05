from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.granularity import Granularity
from athenian.api.models.web.jira_filter_with import JIRAFilterWith
from athenian.api.models.web.jira_metric_id import JIRAMetricID
from athenian.api.models.web.quantiles import validate_quantiles


class JIRAMetricsRequest(Model):
    """Request body of `/metrics/jira`."""

    openapi_types = {
        "account": int,
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "priorities": List[str],
        "types": List[str],
        "with_": List[JIRAFilterWith],
        "metrics": List[str],
        "quantiles": List[float],
        "granularities": List[str],
        "exclude_inactive": bool,
        "labels_include": List[str],
        "labels_exclude": List[str],
        "epics": List[str],
        "group_by_jira_label": Optional[bool],
    }

    attribute_map = {
        "account": "account",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "priorities": "priorities",
        "types": "types",
        "with_": "with",
        "metrics": "metrics",
        "quantiles": "quantiles",
        "granularities": "granularities",
        "exclude_inactive": "exclude_inactive",
        "labels_include": "labels_include",
        "labels_exclude": "labels_exclude",
        "epics": "epics",
        "group_by_jira_label": "group_by_jira_label",
    }

    def __init__(
        self,
        account: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        priorities: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        with_: Optional[List[JIRAFilterWith]] = None,
        metrics: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        granularities: Optional[List[str]] = None,
        exclude_inactive: Optional[bool] = None,
        labels_include: Optional[List[str]] = None,
        labels_exclude: Optional[List[str]] = None,
        epics: Optional[List[str]] = None,
        group_by_jira_label: Optional[bool] = None,
    ):
        """JIRAMetricsRequest - a model defined in OpenAPI

        :param account: The account of this JIRAMetricsRequest.
        :param date_from: The date_from of this JIRAMetricsRequest.
        :param date_to: The date_to of this JIRAMetricsRequest.
        :param timezone: The timezone of this JIRAMetricsRequest.
        :param priorities: The priorities of this JIRAMetricsRequest.
        :param types: The types of this JIRAMetricsRequest.
        :param with_: The with of this JIRAMetricsRequest.
        :param metrics: The metrics of this JIRAMetricsRequest.
        :param quantiles: The quantiles of this JIRAMetricsRequest.
        :param granularities: The granularities of this JIRAMetricsRequest.
        :param labels_include: The labels_include of this JIRAMetricsRequest.
        :param labels_exclude: The labels_exclude of this JIRAMetricsRequest.
        :param epics: The epics of this JIRAMetricsRequest.
        :param group_by_jira_label: The group_by_jira_label of this JIRAMetricsRequest.
        """
        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._priorities = priorities
        self._types = types
        self._with_ = with_
        self._metrics = metrics
        self._quantiles = quantiles
        self._granularities = granularities
        self._exclude_inactive = exclude_inactive
        self._labels_include = labels_include
        self._labels_exclude = labels_exclude
        self._epics = epics
        self._group_by_jira_label = group_by_jira_label

    @property
    def account(self) -> int:
        """Gets the account of this JIRAMetricsRequest.

        Session account ID.

        :return: The account of this JIRAMetricsRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this JIRAMetricsRequest.

        Session account ID.

        :param account: The account of this JIRAMetricsRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def date_from(self) -> date:
        """Gets the date_from of this JIRAMetricsRequest.

        Issues must be resolved after this date.

        :return: The date_from of this JIRAMetricsRequest.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date):
        """Sets the date_from of this JIRAMetricsRequest.

        Issues must be resolved after this date.

        :param date_from: The date_from of this JIRAMetricsRequest.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this JIRAMetricsRequest.

        Issues must be created before this date.

        :return: The date_to of this JIRAMetricsRequest.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date):
        """Sets the date_to of this JIRAMetricsRequest.

        Issues must be created before this date.

        :param date_to: The date_to of this JIRAMetricsRequest.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def timezone(self) -> int:
        """Gets the timezone of this JIRAMetricsRequest.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :return: The timezone of this JIRAMetricsRequest.
        """
        return self._timezone

    @timezone.setter
    def timezone(self, timezone: int):
        """Sets the timezone of this JIRAMetricsRequest.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :param timezone: The timezone of this JIRAMetricsRequest.
        """
        if timezone is not None and timezone > 720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value less than or equal to `720`")
        if timezone is not None and timezone < -720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value greater than or equal to `-720`")

        self._timezone = timezone

    @property
    def priorities(self) -> List[str]:
        """Gets the priorities of this JIRAMetricsRequest.

        Selected issue priorities.

        :return: The priorities of this JIRAMetricsRequest.
        """
        return self._priorities

    @priorities.setter
    def priorities(self, priorities: List[str]):
        """Sets the priorities of this JIRAMetricsRequest.

        Selected issue priorities.

        :param priorities: The priorities of this JIRAMetricsRequest.
        """
        self._priorities = priorities

    @property
    def types(self) -> List[str]:
        """Gets the types of this JIRAMetricsRequest.

        Selected issue types.

        :return: The types of this JIRAMetricsRequest.
        """
        return self._types

    @types.setter
    def types(self, types: List[str]):
        """Sets the types of this JIRAMetricsRequest.

        Selected issue types.

        :param types: The types of this JIRAMetricsRequest.
        """
        self._types = types

    @property
    def with_(self) -> Optional[List[JIRAFilterWith]]:
        """Gets the with of this JIRAMetricsRequest.

        Groups of issue participants. The metrics will be calculated for each group.

        :return: The with of this JIRAMetricsRequest.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[List[JIRAFilterWith]]):
        """Sets the with of this JIRAMetricsRequest.

        Groups of issue participants. The metrics will be calculated for each group.

        :param with_: The with of this JIRAMetricsRequest.
        """
        self._with_ = with_

    @property
    def metrics(self) -> List[str]:
        """Gets the metrics of this JIRAMetricsRequest.

        List of measured metrics.

        :return: The metrics of this JIRAMetricsRequest.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[str]):
        """Sets the metrics of this JIRAMetricsRequest.

        List of measured metrics.

        :param metrics: The metrics of this JIRAMetricsRequest.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for i, metric in enumerate(metrics):
            if metric not in JIRAMetricID:
                raise ValueError("metrics[%d] is not one of %s" % (i + 1, list(JIRAMetricID)))
        self._metrics = metrics

    @property
    def quantiles(self) -> Optional[List[float]]:
        """Gets the quantiles of this JIRAMetricsRequest.

        Cut the distributions at certain quantiles. The default is [0, 1].

        :return: The quantiles of this JIRAMetricsRequest.
        """
        return self._quantiles

    @quantiles.setter
    def quantiles(self, quantiles: Optional[List[float]]):
        """Sets the quantiles of this JIRAMetricsRequest.

        Cut the distributions at certain quantiles. The default is [0, 1].

        :param quantiles: The quantiles of this JIRAMetricsRequest.
        """
        if quantiles is None:
            self._quantiles = None
            return
        validate_quantiles(quantiles)
        self._quantiles = quantiles

    @property
    def granularities(self) -> List[str]:
        """Gets the granularities of this JIRAMetricsRequest.

        Splits of the specified time range `[date_from, date_to)`.

        :return: The granularities of this JIRAMetricsRequest.
        """
        return self._granularities

    @granularities.setter
    def granularities(self, granularities: List[str]):
        """Sets the granularities of this JIRAMetricsRequest.

        Splits of the specified time range `[date_from, date_to)`.

        :param granularities: The granularities of this JIRAMetricsRequest.
        """
        if granularities is None:
            raise ValueError("Invalid value for `granularities`, must not be `None`")
        for i, g in enumerate(granularities):
            if not Granularity.format.match(g):
                raise ValueError(
                    'Invalid value for `granularity[%d]`: "%s"` does not match /%s/' %
                    (i, g, Granularity.format.pattern))

        self._granularities = granularities

    @property
    def exclude_inactive(self) -> bool:
        """Gets the exclude_inactive of this JIRAMetricsRequest.

        Value indicating whether issues with the last update older than
        `date_from` should be ignored.

        :return: The exclude_inactive of this JIRAMetricsRequest.
        """
        return self._exclude_inactive

    @exclude_inactive.setter
    def exclude_inactive(self, exclude_inactive: bool):
        """Sets the exclude_inactive of this JIRAMetricsRequest.

        Value indicating whether issues with the last update older than
        `date_from` should be ignored.

        :param exclude_inactive: The exclude_inactive of this JIRAMetricsRequest.
        """
        self._exclude_inactive = exclude_inactive

    @property
    def labels_include(self) -> Optional[List[str]]:
        """Gets the labels_include of this JIRAMetricsRequest.

        JIRA issues must contain at least one label from the list.
        Several labels may be concatenated by a comma `,` so that all of them
        are required.

        :return: The labels_include of this JIRAMetricsRequest.
        """
        return self._labels_include

    @labels_include.setter
    def labels_include(self, labels_include: Optional[List[str]]):
        """Sets the labels_include of this JIRAMetricsRequest.

        JIRA issues must contain at least one label from the list.
        Several labels may be concatenated by a comma `,` so that all of them
        are required.

        :param labels_include: The labels_include of this JIRAMetricsRequest.
        """
        self._labels_include = labels_include

    @property
    def labels_exclude(self) -> Optional[List[str]]:
        """Gets the labels_exclude of this JIRAMetricsRequest.

        JIRA issues may not contain labels from this list.

        :return: The labels_exclude of this JIRAMetricsRequest.
        """
        return self._labels_exclude

    @labels_exclude.setter
    def labels_exclude(self, labels_exclude: Optional[List[str]]):
        """Sets the labels_exclude of this JIRAMetricsRequest.

        JIRA issues may not contain labels from this list.

        :param labels_exclude: The labels_exclude of this JIRAMetricsRequest.
        """
        self._labels_exclude = labels_exclude

    @property
    def epics(self) -> Optional[List[str]]:
        """Gets the epics of this JIRAMetricsRequest.

        JIRA issues must be attached to any of the epic IDs from this list.

        :return: The epics of this JIRAMetricsRequest.
        """
        return self._epics

    @epics.setter
    def epics(self, epics: Optional[List[str]]):
        """Sets the epics of this JIRAMetricsRequest.

        JIRA issues must be attached to any of the epic IDs from this list.

        :param epics: The epics of this JIRAMetricsRequest.
        """
        self._epics = epics

    @property
    def group_by_jira_label(self) -> Optional[bool]:
        """Gets the group_by_jira_label of this JIRAMetricsRequest.

        Value indicating whether the metrics should be grouped by assigned JIRA issue label.

        :return: The group_by_jira_label of this JIRAMetricsRequest.
        """
        return self._group_by_jira_label

    @group_by_jira_label.setter
    def group_by_jira_label(self, group_by_jira_label: Optional[bool]):
        """Sets the group_by_jira_label of this JIRAMetricsRequest.

        Value indicating whether the metrics should be grouped by assigned JIRA issue label.

        :param group_by_jira_label: The group_by_jira_label of this JIRAMetricsRequest.
        """
        self._group_by_jira_label = group_by_jira_label
