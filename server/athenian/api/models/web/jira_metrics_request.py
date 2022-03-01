from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_metrics_properties import CommonMetricsProperties
from athenian.api.models.web.filter_jira_common import FilterJIRACommon
from athenian.api.models.web.jira_filter_with import JIRAFilterWith
from athenian.api.models.web.jira_metric_id import JIRAMetricID


class JIRAMetricsRequestSpecials(Model, sealed=False):
    """Request body of `/metrics/jira`."""

    openapi_types = {
        "with_": List[JIRAFilterWith],
        "metrics": List[str],
        "epics": List[str],
        "group_by_jira_label": Optional[bool],
    }

    attribute_map = {
        "with_": "with",
        "metrics": "metrics",
        "epics": "epics",
        "group_by_jira_label": "group_by_jira_label",
    }

    def __init__(
        self,
        with_: Optional[List[JIRAFilterWith]] = None,
        metrics: Optional[List[str]] = None,
        epics: Optional[List[str]] = None,
        group_by_jira_label: Optional[bool] = None,
    ):
        """JIRAMetricsRequest - a model defined in OpenAPI

        :param with_: The with of this JIRAMetricsRequest.
        :param metrics: The metrics of this JIRAMetricsRequest.
        :param epics: The epics of this JIRAMetricsRequest.
        :param group_by_jira_label: The group_by_jira_label of this JIRAMetricsRequest.
        """
        self._with_ = with_
        self._metrics = metrics
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


JIRAMetricsRequest = AllOf(FilterJIRACommon,
                           CommonMetricsProperties,
                           JIRAMetricsRequestSpecials,
                           name="JIRAMetricsRequest",
                           module=__name__)
