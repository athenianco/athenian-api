from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_metrics_properties import CommonMetricsProperties
from athenian.api.models.web.filter_jira_common import FilterJIRACommon
from athenian.api.models.web.jira_filter_with import JIRAFilterWith
from athenian.api.models.web.jira_metric_id import JIRAMetricID


class JIRAMetricsRequestSpecials(Model, sealed=False):
    """Request body of `/metrics/jira`."""

    with_: (Optional[list[JIRAFilterWith]], "with")
    metrics: list[str]
    epics: Optional[list[str]]
    group_by_jira_label: Optional[bool]

    def validate_metrics(self, metrics: list[str]) -> list[str]:
        """Sets the metrics of this JIRAMetricsRequest.

        List of measured metrics.

        :param metrics: The metrics of this JIRAMetricsRequest.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for i, metric in enumerate(metrics):
            if metric not in JIRAMetricID:
                raise ValueError("metrics[%d] is not one of %s" % (i + 1, list(JIRAMetricID)))
        return metrics


JIRAMetricsRequest = AllOf(
    FilterJIRACommon,
    CommonMetricsProperties,
    JIRAMetricsRequestSpecials,
    name="JIRAMetricsRequest",
    module=__name__,
)
