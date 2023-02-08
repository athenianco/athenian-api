from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import CalculatedLinearMetricValues
from athenian.api.models.web.granularity import GranularityMixin
from athenian.api.models.web.jira_filter import JIRAFilter
from athenian.api.models.web.release_metric_id import ReleaseMetricID
from athenian.api.models.web.release_with import ReleaseWith


class CalculatedReleaseMetric(Model, GranularityMixin):
    """Response from `/metrics/releases`."""

    for_: (Optional[list[str]], "for")
    with_: (Optional[ReleaseWith], "with")
    matches: dict[str, str]
    metrics: list[str]
    granularity: str
    values: list[CalculatedLinearMetricValues]
    jira: Optional[JIRAFilter]

    def validate_metrics(self, metrics: list[str]) -> list[str]:
        """Sets the metrics of this CalculatedReleaseMetric.

        :param metrics: The metrics of this CalculatedReleaseMetric.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for i, metric in enumerate(metrics):
            if metric not in ReleaseMetricID:
                raise ValueError(
                    "`metrics[%d]` = '%s' must be one of %s" % (i, metric, list(ReleaseMetricID)),
                )

        return metrics
