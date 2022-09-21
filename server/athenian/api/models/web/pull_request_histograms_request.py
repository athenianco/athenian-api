from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import QuantilesMixin
from athenian.api.models.web.for_set_pull_requests import ForSetPullRequests
from athenian.api.models.web.pull_request_histogram_definition import (
    PullRequestHistogramDefinition,
)


class _PullRequestHistogramsRequest(Model, QuantilesMixin, sealed=False):
    """Request of `/histograms/pull_requests`."""

    for_: (list[ForSetPullRequests], "for")
    histograms: list[PullRequestHistogramDefinition]
    exclude_inactive: bool
    quantiles: list[float]
    fresh: Optional[bool]


PullRequestHistogramsRequest = AllOf(
    _PullRequestHistogramsRequest,
    CommonFilterProperties,
    name="PullRequestHistogramsRequest",
    module=__name__,
)
