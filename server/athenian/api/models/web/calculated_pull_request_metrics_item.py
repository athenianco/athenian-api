from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import CalculatedLinearMetricValues
from athenian.api.models.web.for_set_pull_requests import ForSetPullRequests
from athenian.api.models.web.granularity import GranularityMixin


class CalculatedPullRequestMetricsItem(Model, GranularityMixin):
    """Series of calculated metrics for a specific set of repositories and developers."""

    attribute_types = {
        "for_": ForSetPullRequests,
        "granularity": str,
        "values": List[CalculatedLinearMetricValues],
    }

    attribute_map = {"for_": "for", "granularity": "granularity", "values": "values"}

    def __init__(
        self,
        for_: Optional[ForSetPullRequests] = None,
        granularity: Optional[str] = None,
        values: Optional[List[CalculatedLinearMetricValues]] = None,
    ):
        """CalculatedPullRequestMetricsItem - a model defined in OpenAPI."""
        self._for_ = for_
        self._granularity = granularity
        self._values = values

    @property
    def for_(self) -> ForSetPullRequests:
        """Gets the for_ of this CalculatedPullRequestMetricsItem."""
        return self._for_

    @for_.setter
    def for_(self, for_: ForSetPullRequests) -> None:
        """Sets the for_ of this CalculatedPullRequestMetricsItem."""
        if for_ is None:
            raise ValueError("Invalid value for `for`, must not be `None`")

        self._for_ = for_

    @property
    def values(self) -> List[CalculatedLinearMetricValues]:
        """Gets the values of this CalculatedPullRequestMetricsItem."""
        return self._values

    @values.setter
    def values(self, values: List[CalculatedLinearMetricValues]) -> None:
        """Sets the values of this CalculatedPullRequestMetricsItem."""
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values
