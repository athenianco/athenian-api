from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_developer_metrics_item import (
    CalculatedDeveloperMetricsItem,
)
from athenian.api.models.web.granularity import Granularity


class CalculatedDeveloperMetrics(Model):
    """The dates start from `date_from` and end earlier or equal to `date_to`."""

    calculated: List[CalculatedDeveloperMetricsItem]
    metrics: List[str]
    date_from: date
    date_to: date
    timezone: Optional[int]
    granularities: List[str]

    def validate_timezone(self, timezone: int) -> int:
        """Sets the timezone of this CalculatedDeveloperMetrics.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :param timezone: The timezone of this CalculatedDeveloperMetrics.
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

    def validate_granularities(self, granularities: list[str]) -> list[str]:
        """Sets the granularities of this CalculatedDeveloperMetrics.

        :param granularities: The granularities of this CalculatedDeveloperMetrics.
        """
        if granularities is None:
            raise ValueError("Invalid value for `granularities`, must not be `None`")
        for i, g in enumerate(granularities):
            if not Granularity.format.match(g):
                raise ValueError(
                    'Invalid value for `granularity[%d]`: "%s"` does not match /%s/'
                    % (i, g, Granularity.format.pattern),
                )

        return granularities
