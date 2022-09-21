from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.code_check_metric_id import CodeCheckMetricID
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import CommonMetricsProperties
from athenian.api.models.web.for_set_code_checks import ForSetCodeChecks


class _CodeCheckMetricsRequest(Model, sealed=False):
    """Request for calculating metrics on top of code check runs (CI) data."""

    for_: (list[ForSetCodeChecks], "for")
    metrics: list[CodeCheckMetricID]
    split_by_check_runs: Optional[bool]


CodeCheckMetricsRequest = AllOf(
    _CodeCheckMetricsRequest,
    CommonFilterProperties,
    CommonMetricsProperties,
    name="CodeCheckMetricsRequest",
    module=__name__,
)
