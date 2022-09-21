from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import CommonMetricsProperties
from athenian.api.models.web.for_set_pull_requests import ForSetPullRequests
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID


class _PullRequestMetricsRequest(Model, sealed=False):
    """Request for calculating metrics on top of pull requests data."""

    for_: (list[ForSetPullRequests], "for")
    metrics: list[PullRequestMetricID]
    exclude_inactive: bool
    fresh: Optional[bool]


PullRequestMetricsRequest = AllOf(
    _PullRequestMetricsRequest,
    CommonFilterProperties,
    CommonMetricsProperties,
    name="PullRequestMetricsRequest",
    module=__name__,
)
