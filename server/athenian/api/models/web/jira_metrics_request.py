from datetime import date
from typing import List, Optional

from athenian.api.models.web import Granularity
from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_metric_id import JIRAMetricID


class JIRAMetricsRequest(Model):
    """Request body of `/metrics/jira`."""

    openapi_types = {
        "account": int,
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "priorities": List[str],
        "assignees": List[str],
        "reporters": List[str],
        "commenters": List[str],
        "stakeholders": List[str],
        "metrics": List[JIRAMetricID],
        "quantiles": List[float],
        "granularities": List[str],
    }

    attribute_map = {
        "account": "account",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "priorities": "priorities",
        "assignees": "assignees",
        "reporters": "reporters",
        "commenters": "commenters",
        "stakeholders": "stakeholders",
        "metrics": "metrics",
        "quantiles": "quantiles",
        "granularities": "granularities",
    }

    def __init__(
        self,
        account: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        priorities: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
        reporters: Optional[List[str]] = None,
        commenters: Optional[List[str]] = None,
        stakeholders: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        granularities: Optional[List[str]] = None,
    ):
        """JIRAMetricsRequest - a model defined in OpenAPI

        :param account: The account of this JIRAMetricsRequest.
        :param date_from: The date_from of this JIRAMetricsRequest.
        :param date_to: The date_to of this JIRAMetricsRequest.
        :param timezone: The timezone of this JIRAMetricsRequest.
        :param priorities: The priorities of this JIRAMetricsRequest.
        :param assignees: The assignees of this JIRAMetricsRequest.
        :param reporters: The reporters of this JIRAMetricsRequest.
        :param commenters: The commenters of this JIRAMetricsRequest.
        :param stakeholders: The stakeholders of this JIRAMetricsRequest.
        :param metrics: The metrics of this JIRAMetricsRequest.
        :param quantiles: The quantiles of this JIRAMetricsRequest.
        :param granularities: The granularities of this JIRAMetricsRequest.
        """
        self._account = account
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._priorities = priorities
        self._assignees = assignees
        self._reporters = reporters
        self._commenters = commenters
        self._stakeholders = stakeholders
        self._metrics = metrics
        self._quantiles = quantiles
        self._granularities = granularities

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
    def assignees(self) -> List[str]:
        """Gets the assignees of this JIRAMetricsRequest.

        Selected issue assignee users.

        :return: The assignees of this JIRAMetricsRequest.
        """
        return self._assignees

    @assignees.setter
    def assignees(self, assignees: List[str]):
        """Sets the assignees of this JIRAMetricsRequest.

        Selected issue assignee users.

        :param assignees: The assignees of this JIRAMetricsRequest.
        """
        self._assignees = assignees

    @property
    def reporters(self) -> List[str]:
        """Gets the reporters of this JIRAMetricsRequest.

        Selected issue reporter users.

        :return: The reporters of this JIRAMetricsRequest.
        """
        return self._reporters

    @reporters.setter
    def reporters(self, reporters: List[str]):
        """Sets the reporters of this JIRAMetricsRequest.

        Selected issue reporter users.

        :param reporters: The reporters of this JIRAMetricsRequest.
        """
        self._reporters = reporters

    @property
    def commenters(self) -> List[str]:
        """Gets the commenters of this JIRAMetricsRequest.

        Selected issue commenter users.

        :return: The commenters of this JIRAMetricsRequest.
        """
        return self._commenters

    @commenters.setter
    def commenters(self, commenters: List[str]):
        """Sets the commenters of this JIRAMetricsRequest.

        Selected issue commenter users.

        :param commenters: The commenters of this JIRAMetricsRequest.
        """
        self._commenters = commenters

    @property
    def stakeholders(self) -> List[str]:
        """Gets the stakeholders of this JIRAMetricsRequest.

        Selected issue stakeholder users.

        :return: The stakeholders of this JIRAMetricsRequest.
        """
        return self._stakeholders

    @stakeholders.setter
    def stakeholders(self, stakeholders: List[str]):
        """Sets the stakeholders of this JIRAMetricsRequest.

        Selected issue stakeholder users.

        :param stakeholders: The stakeholders of this JIRAMetricsRequest.
        """
        self._stakeholders = stakeholders

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
    def quantiles(self) -> List[float]:
        """Gets the quantiles of this JIRAMetricsRequest.

        Cut the distributions at certain quantiles. The default is [0, 1].

        :return: The quantiles of this JIRAMetricsRequest.
        """
        return self._quantiles

    @quantiles.setter
    def quantiles(self, quantiles: List[float]):
        """Sets the quantiles of this JIRAMetricsRequest.

        Cut the distributions at certain quantiles. The default is [0, 1].

        :param quantiles: The quantiles of this JIRAMetricsRequest.
        """
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
