from typing import Optional

from athenian.api.models.web.calculated_pull_request_metrics_item import (
    CalculatedPullRequestMetricsItem,
)
from athenian.api.models.web.common_filter_properties import TimeFilterProperties
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID


class CalculatedPullRequestMetrics(TimeFilterProperties):
    """This class is auto generated by OpenAPI Generator (https://openapi-generator.tech)."""

    calculated: list[CalculatedPullRequestMetricsItem]
    metrics: list[str]
    granularities: list[str]
    quantiles: Optional[list[float]]
    exclude_inactive: bool

    def validate_metrics(self, metrics: list[str]) -> list[str]:
        """Sets the metrics of this CalculatedPullRequestMetrics.

        Repeats `PullRequestMetricsRequest.metrics`.

        :param metrics: The metrics of this CalculatedPullRequestMetrics.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for m in metrics:
            if m not in PullRequestMetricID:
                raise ValueError(
                    'Invalid value for `metrics`: "%s" must be one of %s' % m,
                    list(PullRequestMetricID),
                )

        return metrics
