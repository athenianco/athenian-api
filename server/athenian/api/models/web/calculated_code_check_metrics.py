from typing import Optional

from athenian.api.models.web.calculated_code_check_metrics_item import (
    CalculatedCodeCheckMetricsItem,
)
from athenian.api.models.web.code_check_metric_id import CodeCheckMetricID
from athenian.api.models.web.common_filter_properties import TimeFilterProperties


class CalculatedCodeCheckMetrics(TimeFilterProperties):
    """Response from `/metrics/code_checks`."""

    calculated: list[CalculatedCodeCheckMetricsItem]
    metrics: list[CodeCheckMetricID]
    granularities: list[str]
    quantiles: Optional[list[float]]
    split_by_check_runs: Optional[bool]
