from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.granularity import Granularity
from athenian.api.models.web.quantiles import validate_quantiles


class GranularitiesMixin:
    """Implement `granularities` property."""

    def validate_granularities(self, granularities: list[str]) -> list[str]:
        """Sets the granularities of this CommonMetricsProperties.

        Splits of the specified time range `[date_from, date_to)`.

        :param granularities: The granularities of this CommonMetricsProperties.
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


class QuantilesMixin:
    """Implement `quantiles` property."""

    def validate_quantiles(self, quantiles: Optional[list[float]]) -> Optional[list[float]]:
        """Sets the quantiles of this CommonMetricsProperties.

        :param quantiles: The quantiles of this CommonMetricsProperties.
        """
        if quantiles is None:
            return
        validate_quantiles(quantiles)
        return quantiles


class CommonMetricsProperties(Model, GranularitiesMixin, QuantilesMixin, sealed=False):
    """Define `account`, `date_from`, `date_to`, and `timezone` properties."""

    granularities: List[str]
    quantiles: Optional[List[float]]
