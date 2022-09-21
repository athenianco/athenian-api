from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import CommonMetricsProperties
from athenian.api.models.web.for_set_common import CommonPullRequestFilters
from athenian.api.models.web.release_metric_id import ReleaseMetricID
from athenian.api.models.web.release_with import ReleaseWith


class _ReleaseMetricsRequest(Model, sealed=False):
    """Request of `/metrics/releases` to calculate metrics on releases."""

    for_: (List[List[str]], "for")
    with_: (Optional[List[ReleaseWith]], "with")
    metrics: List[str]

    def validate_metrics(self, metrics: list[str]) -> list[str]:
        """Sets the metrics of this ReleaseMetricsRequest.

        List of desired release metrics.

        :param metrics: The metrics of this ReleaseMetricsRequest.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for i, metric in enumerate(metrics):
            if metric not in ReleaseMetricID:
                raise ValueError(
                    "`metrics[%d]` = '%s' must be one of %s" % (i, metric, list(ReleaseMetricID)),
                )

        return metrics


ReleaseMetricsRequest = AllOf(
    _ReleaseMetricsRequest,
    CommonFilterProperties,
    CommonMetricsProperties,
    CommonPullRequestFilters,
    name="ReleaseMetricsRequest",
    module=__name__,
)
