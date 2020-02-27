from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_metric_values import CalculatedMetricValues
from athenian.api.models.web.for_set import ForSet


class CalculatedMetric(Model):
    """Series of calculated metrics for a specific set of repositories and developers."""

    def __init__(
        self, for_: Optional[ForSet] = None, values: Optional[List[CalculatedMetricValues]] = None,
    ):
        """CalculatedMetric - a model defined in OpenAPI

        :param for_: The for_ of this CalculatedMetric.
        :param values: The values of this CalculatedMetric.
        """
        self.openapi_types = {"for_": ForSet, "values": List[CalculatedMetricValues]}

        self.attribute_map = {"for_": "for", "values": "values"}

        self._for_ = for_
        self._values = values

    @property
    def for_(self):
        """Gets the for_ of this CalculatedMetric.

        :return: The for of this CalculatedMetric.
        :rtype: ForSet
        """
        return self._for_

    @for_.setter
    def for_(self, for_):
        """Sets the for_ of this CalculatedMetric.

        :param for_: The for of this CalculatedMetric.
        :type for_: ForSet
        """
        if for_ is None:
            raise ValueError("Invalid value for `for`, must not be `None`")

        self._for_ = for_

    @property
    def values(self):
        """Gets the values of this CalculatedMetric.

        :return: The values of this CalculatedMetric.
        :rtype: List[CalculatedMetricValues]
        """
        return self._values

    @values.setter
    def values(self, values):
        """Sets the values of this CalculatedMetric.

        :param values: The values of this CalculatedMetric.
        :type values: List[CalculatedMetricValues]
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values
