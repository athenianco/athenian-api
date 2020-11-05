from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import \
    CalculatedLinearMetricValues
from athenian.api.models.web.jira_metrics_request_with import JIRAMetricsRequestWith


class CalculatedJIRAMetricValues(Model):
    """Calculated JIRA metrics for a specific granularity."""

    openapi_types = {
        "granularity": str,
        "with_": Optional[JIRAMetricsRequestWith],
        "values": List[CalculatedLinearMetricValues],
    }

    attribute_map = {
        "granularity": "granularity",
        "with_": "with",
        "values": "values",
    }

    def __init__(self,
                 granularity: Optional[str] = None,
                 with_: Optional[JIRAMetricsRequestWith] = None,
                 values: Optional[List[CalculatedLinearMetricValues]] = None):
        """CalculatedJIRAMetricValues - a model defined in OpenAPI

        :param granularity: The granularity of this CalculatedJIRAMetricValues.
        :param with_: The with of this CalculatedJIRAMetricValues.
        :param values: The values of this CalculatedJIRAMetricValues.
        """
        self._granularity = granularity
        self._with_ = with_
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
    def with_(self) -> Optional[List[JIRAMetricsRequestWith]]:
        """Gets the with of this CalculatedJIRAMetricValues.

        Groups of issue participants. The metrics will be calculated for each group.

        :return: The with of this CalculatedJIRAMetricValues.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[List[JIRAMetricsRequestWith]]):
        """Sets the with of this CalculatedJIRAMetricValues.

        Groups of issue participants. The metrics will be calculated for each group.

        :param with_: The with of this CalculatedJIRAMetricValues.
        """
        self._with_ = with_

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
