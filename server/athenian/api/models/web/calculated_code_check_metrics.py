from datetime import date
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_code_check_metrics_item import (
    CalculatedCodeCheckMetricsItem,
)
from athenian.api.models.web.code_check_metric_id import CodeCheckMetricID


class CalculatedCodeCheckMetrics(Model):
    """Response from `/metrics/code_checks`."""

    calculated: list[CalculatedCodeCheckMetricsItem]
    metrics: list[CodeCheckMetricID]
    date_from: date
    date_to: date
    timezone: int
    granularities: list[str]
    quantiles: Optional[list[float]]
    split_by_check_runs: Optional[bool]

    def validate_timezone(self, timezone: int) -> int:
        """Sets the timezone of this CalculatedCodeCheckMetrics.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :param timezone: The timezone of this CalculatedCodeCheckMetrics.
        """
        if timezone is not None and timezone > 720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value less than or equal to `720`",
            )
        if timezone is not None and timezone < -720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value greater than or equal to `-720`",
            )

        return timezone
