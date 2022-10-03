from athenian.api.models.web.calculated_developer_metrics_item import (
    CalculatedDeveloperMetricsItem,
)
from athenian.api.models.web.common_filter_properties import TimeFilterProperties
from athenian.api.models.web.granularity import Granularity


class CalculatedDeveloperMetrics(TimeFilterProperties):
    """The dates start from `date_from` and end earlier or equal to `date_to`."""

    calculated: list[CalculatedDeveloperMetricsItem]
    metrics: list[str]
    granularities: list[str]

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
