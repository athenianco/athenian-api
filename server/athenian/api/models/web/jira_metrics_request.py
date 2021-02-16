from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.filter_jira_common import FilterJIRACommon
from athenian.api.models.web.granularity import Granularity
from athenian.api.models.web.jira_filter_with import JIRAFilterWith
from athenian.api.models.web.jira_metric_id import JIRAMetricID
from athenian.api.models.web.quantiles import validate_quantiles


class JIRAMetricsRequestSpecials(Model):
    """Request body of `/metrics/jira`."""

    openapi_types = {
        "date_from": date,
        "date_to": date,
        "with_": List[JIRAFilterWith],
        "metrics": List[str],
        "quantiles": List[float],
        "granularities": List[str],
        "epics": List[str],
        "group_by_jira_label": Optional[bool],
    }

    attribute_map = {
        "date_from": "date_from",
        "date_to": "date_to",
        "with_": "with",
        "metrics": "metrics",
        "quantiles": "quantiles",
        "granularities": "granularities",
        "epics": "epics",
        "group_by_jira_label": "group_by_jira_label",
    }

    __enable_slots__ = False

    def __init__(
        self,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        with_: Optional[List[JIRAFilterWith]] = None,
        metrics: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        granularities: Optional[List[str]] = None,
        epics: Optional[List[str]] = None,
        group_by_jira_label: Optional[bool] = None,
    ):
        """JIRAMetricsRequest - a model defined in OpenAPI

        :param date_from: The date_from of this JIRAMetricsRequest.
        :param date_to: The date_to of this JIRAMetricsRequest.
        :param with_: The with of this JIRAMetricsRequest.
        :param metrics: The metrics of this JIRAMetricsRequest.
        :param quantiles: The quantiles of this JIRAMetricsRequest.
        :param granularities: The granularities of this JIRAMetricsRequest.
        :param epics: The epics of this JIRAMetricsRequest.
        :param group_by_jira_label: The group_by_jira_label of this JIRAMetricsRequest.
        """
        self._date_from = date_from
        self._date_to = date_to
        self._with_ = with_
        self._metrics = metrics
        self._quantiles = quantiles
        self._granularities = granularities
        self._epics = epics
        self._group_by_jira_label = group_by_jira_label

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


JIRAMetricsRequest = AllOf(FilterJIRACommon, JIRAMetricsRequestSpecials,
                           name="JIRAMetricsRequest", module=__name__)
