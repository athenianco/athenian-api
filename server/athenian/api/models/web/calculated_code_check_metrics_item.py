from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.calculated_linear_metric_values import CalculatedLinearMetricValues
from athenian.api.models.web.for_set_code_checks import _CalculatedCodeCheckCommon
from athenian.api.models.web.granularity import GranularityMixin


class _CalculatedCodeCheckMetricsItem(Model, GranularityMixin, sealed=False):
    """Series of calculated metrics for a specific set of repositories and commit authors."""

    granularity: str
    values: list[CalculatedLinearMetricValues]


CalculatedCodeCheckMetricsItem = AllOf(
    _CalculatedCodeCheckMetricsItem,
    _CalculatedCodeCheckCommon,
    name="CalculatedCodeCheckMetricsItem",
    module=__name__,
)
