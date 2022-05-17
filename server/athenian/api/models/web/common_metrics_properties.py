from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.granularity import Granularity
from athenian.api.models.web.quantiles import validate_quantiles


class GranularitiesMixin:
    """Implement `granularities` property."""

    @property
    def granularities(self) -> List[str]:
        """Gets the granularities of this CommonMetricsProperties.

        Splits of the specified time range `[date_from, date_to)`.

        :return: The granularities of this CommonMetricsProperties.
        """
        return self._granularities

    @granularities.setter
    def granularities(self, granularities: List[str]):
        """Sets the granularities of this CommonMetricsProperties.

        Splits of the specified time range `[date_from, date_to)`.

        :param granularities: The granularities of this CommonMetricsProperties.
        """
        if granularities is None:
            raise ValueError("Invalid value for `granularities`, must not be `None`")
        for i, g in enumerate(granularities):
            if not Granularity.format.match(g):
                raise ValueError(
                    'Invalid value for `granularity[%d]`: "%s"` does not match /%s/' %
                    (i, g, Granularity.format.pattern))

        self._granularities = granularities


class QuantilesMixin:
    """Implement `quantiles` property."""

    @property
    def quantiles(self) -> Optional[List[float]]:
        """Gets the quantiles of this CommonMetricsProperties.

        :return: The quantiles of this CommonMetricsProperties.
        """
        return self._quantiles

    @quantiles.setter
    def quantiles(self, quantiles: Optional[List[float]]):
        """Sets the quantiles of this CommonMetricsProperties.

        :param quantiles: The quantiles of this CommonMetricsProperties.
        """
        if quantiles is None:
            self._quantiles = None
            return
        validate_quantiles(quantiles)
        self._quantiles = quantiles


class CommonMetricsProperties(Model, GranularitiesMixin, QuantilesMixin, sealed=False):
    """Define `account`, `date_from`, `date_to`, and `timezone` properties."""

    attribute_types = {
        "granularities": List[str],
        "quantiles": Optional[List[float]],
    }

    attribute_map = {
        "granularities": "granularities",
        "quantiles": "quantiles",
    }

    def __init__(
        self,
        granularities: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
    ):
        """CommonFilterProperties - a model defined in OpenAPI

        :param granularities: The granularities of this CommonMetricsProperties.
        :param quantiles: The quantiles of this CommonMetricsProperties.
        """
        self._granularities = granularities
        self._quantiles = quantiles
