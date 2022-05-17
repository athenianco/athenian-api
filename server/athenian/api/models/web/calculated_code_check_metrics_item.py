from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.calculated_linear_metric_values import \
    CalculatedLinearMetricValues
from athenian.api.models.web.for_set_code_checks import _CalculatedCodeCheckCommon
from athenian.api.models.web.granularity import GranularityMixin


class _CalculatedCodeCheckMetricsItem(Model, GranularityMixin, sealed=False):
    """Series of calculated metrics for a specific set of repositories and commit authors."""

    attribute_types = {
        "granularity": str,
        "values": List[CalculatedLinearMetricValues],
    }

    attribute_map = {
        "granularity": "granularity",
        "values": "values",
    }

    def __init__(
        self,
        granularity: Optional[str] = None,
        values: Optional[List[CalculatedLinearMetricValues]] = None,
    ):
        """CalculatedCodeCheckMetricsItem - a model defined in OpenAPI

        :param granularity: The granularity of this CalculatedCodeCheckMetricsItem.
        :param values: The values of this CalculatedCodeCheckMetricsItem.
        """
        self._granularity = granularity
        self._values = values

    @property
    def values(self) -> List[CalculatedLinearMetricValues]:
        """Gets the values of this CalculatedCodeCheckMetricsItem.

        The sequence steps from `date_from` till `date_to` by `granularity`.

        :return: The values of this CalculatedCodeCheckMetricsItem.
        """
        return self._values

    @values.setter
    def values(self, values: List[CalculatedLinearMetricValues]):
        """Sets the values of this CalculatedCodeCheckMetricsItem.

        The sequence steps from `date_from` till `date_to` by `granularity`.

        :param values: The values of this CalculatedCodeCheckMetricsItem.
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values


CalculatedCodeCheckMetricsItem = AllOf(_CalculatedCodeCheckMetricsItem,
                                       _CalculatedCodeCheckCommon,
                                       name="CalculatedCodeCheckMetricsItem",
                                       module=__name__)
