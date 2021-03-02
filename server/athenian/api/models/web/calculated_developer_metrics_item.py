from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import CalculatedLinearMetricValues
from athenian.api.models.web.for_set_developers import ForSetDevelopers
from athenian.api.models.web.granularity import Granularity


class CalculatedDeveloperMetricsItem(Model):
    """
    Measured developer metrics for each `DeveloperMetricsRequest.for`.

    Each repository group maps to a distinct `CalculatedDeveloperMetricsItem`.
    """

    openapi_types = {
        "for_": ForSetDevelopers,
        "granularity": str,
        "values": List[List[CalculatedLinearMetricValues]],
    }

    attribute_map = {
        "for_": "for",
        "granularity": "granularity",
        "values": "values",
    }

    def __init__(
        self,
        for_: Optional[ForSetDevelopers] = None,
        granularity: Optional[str] = None,
        values: Optional[List[List[CalculatedLinearMetricValues]]] = None,
    ):
        """CalculatedDeveloperMetricsItem - a model defined in OpenAPI

        :param for_: The for_ of this CalculatedDeveloperMetricsItem.
        :param granularity: The granularity of this CalculatedDeveloperMetricsItem.
        :param values: The values of this CalculatedDeveloperMetricsItem.
        """
        self._for_ = for_
        self._granularity = granularity
        self._values = values

    @property
    def for_(self) -> ForSetDevelopers:
        """Gets the for_ of this CalculatedDeveloperMetricsItem.

        :return: The for_ of this CalculatedDeveloperMetricsItem.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: ForSetDevelopers):
        """Sets the for_ of this CalculatedDeveloperMetricsItem.

        :param for_: The for_ of this CalculatedDeveloperMetricsItem.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for_`, must not be `None`")

        self._for_ = for_

    @property
    def granularity(self) -> str:
        """Gets the granularity of this CalculatedDeveloperMetricsItem.

        How often the metrics are reported. The value must satisfy the following regular
        expression: /^all|(([1-9]\\d* )?(aligned )?(day|week|month|year))$/. \"all\" produces
        a single interval [`date_from`, `date_to`]. \"aligned week/month/year\" produces
        intervals cut by calendar week/month/year borders, for example, when `date_from` is
        `2020-01-15` and `date_to` is `2020-03-10`, the intervals will be
        `2020-01-15` - `2020-02-01` - `2020-03-01` - `2020-03-10`.

        :return: The granularity of this CalculatedDeveloperMetricsItem.
        """
        return self._granularity

    @granularity.setter
    def granularity(self, granularity: str):
        """Sets the granularity of this CalculatedDeveloperMetricsItem.

        How often the metrics are reported. The value must satisfy the following regular
        expression: /^all|(([1-9]\\d* )?(aligned )?(day|week|month|year))$/. \"all\" produces
        a single interval [`date_from`, `date_to`]. \"aligned week/month/year\" produces
        intervals cut by calendar week/month/year borders, for example, when `date_from` is
        `2020-01-15` and `date_to` is `2020-03-10`, the intervals will be
        `2020-01-15` - `2020-02-01` - `2020-03-01` - `2020-03-10`.

        :param granularity: The granularity of this CalculatedDeveloperMetricsItem.
        """
        if granularity is None:
            raise ValueError("Invalid value for `granularity`, must not be `None`")
        if not Granularity.format.match(granularity):
            raise ValueError(
                'Invalid value for `granularity`: "%s"` does not match /%s/' %
                (granularity, Granularity.format.pattern))

        self._granularity = granularity

    @property
    def values(self) -> List[List[CalculatedLinearMetricValues]]:
        """Gets the values of this CalculatedDeveloperMetricsItem.

        The sequence matches `CalculatedDeveloperMetricsItem.for.developers`.

        :return: The values of this CalculatedDeveloperMetricsItem.
        """
        return self._values

    @values.setter
    def values(self, values: List[List[CalculatedLinearMetricValues]]):
        """Sets the values of this CalculatedDeveloperMetricsItem.

        The sequence matches `CalculatedDeveloperMetricsItem.for.developers`.

        :param values: The values of this CalculatedDeveloperMetricsItem.
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values
