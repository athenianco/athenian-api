from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import CalculatedLinearMetricValues
from athenian.api.models.web.for_set_pull_requests import ForSetPullRequests
from athenian.api.models.web.granularity import GranularityMixin


class CalculatedPullRequestMetricsItem(Model, GranularityMixin):
    """Series of calculated metrics for a specific set of repositories and developers."""

    for_: (ForSetPullRequests, "for")
    granularity: str
    values: list[CalculatedLinearMetricValues]
