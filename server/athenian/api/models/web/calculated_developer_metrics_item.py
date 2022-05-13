from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import CalculatedLinearMetricValues
from athenian.api.models.web.for_set_developers import ForSetDevelopers
from athenian.api.models.web.granularity import GranularityMixin


class CalculatedDeveloperMetricsItem(Model, GranularityMixin):
    """
    Measured developer metrics for each `DeveloperMetricsRequest.for`.

    Each repository group maps to a distinct `CalculatedDeveloperMetricsItem`.
    """

    attribute_types = {
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
