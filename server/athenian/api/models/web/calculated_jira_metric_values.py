from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import \
    CalculatedLinearMetricValues


class CalculatedJIRAMetricValues(Model):
    """Calculated JIRA metrics for a specific granularity."""

    openapi_types = {
        "granularity": str,
        "values": List[CalculatedLinearMetricValues],
    }

    attribute_map = {"granularity": "granularity", "values": "values"}

    def __init__(self,
                 granularity: Optional[str] = None,
                 values: Optional[List[CalculatedLinearMetricValues]] = None):
        """CalculatedJIRAMetricValues - a model defined in OpenAPI

        :param granularity: The granularity of this CalculatedJIRAMetricValues.
        :param values: The values of this CalculatedJIRAMetricValues.
        """
        self._granularity = granularity
        self._values = values

    @property
    def granularity(self) -> str:
        """Gets the granularity of this CalculatedJIRAMetricValues.

        How often the metrics are reported. The value must satisfy the following regular \
        expression: /^(([1-9]\\d* )?(day|week|month|year)|all)$/. \"all\" produces a single \
        interval [`date_from`, `date_to`].

        :return: The granularity of this CalculatedJIRAMetricValues.
        """
        return self._granularity

    @granularity.setter
    def granularity(self, granularity: str):
        """Sets the granularity of this CalculatedJIRAMetricValues.

        How often the metrics are reported. The value must satisfy the following regular
        expression: /^(([1-9]\\d* )?(day|week|month|year)|all)$/. \"all\" produces a single
        interval [`date_from`, `date_to`].

        :param granularity: The granularity of this CalculatedJIRAMetricValues.
        """
        if granularity is None:
            raise ValueError("Invalid value for `granularity`, must not be `None`")

        self._granularity = granularity

    @property
    def values(self) -> List[CalculatedLinearMetricValues]:
        """Gets the values of this CalculatedJIRAMetricValues.

        :return: The values of this CalculatedJIRAMetricValues.
        """
        return self._values

    @values.setter
    def values(self, values: List[CalculatedLinearMetricValues]):
        """Sets the values of this CalculatedJIRAMetricValues.

        :param values: The values of this CalculatedJIRAMetricValues.
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values
