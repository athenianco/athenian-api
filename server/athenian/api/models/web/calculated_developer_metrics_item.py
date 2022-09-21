from typing import List

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import CalculatedLinearMetricValues
from athenian.api.models.web.for_set_developers import ForSetDevelopers
from athenian.api.models.web.granularity import GranularityMixin


class CalculatedDeveloperMetricsItem(Model, GranularityMixin):
    """
    Measured developer metrics for each `DeveloperMetricsRequest.for`.

    Each repository group maps to a distinct `CalculatedDeveloperMetricsItem`.
    """

    for_: (ForSetDevelopers, "for")
    granularity: str
    values: List[List[CalculatedLinearMetricValues]]
