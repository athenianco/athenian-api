from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_pull_request_metric_values import \
    CalculatedPullRequestMetricValues
from athenian.api.models.web.for_set import ForSet


class CalculatedPullRequestMetricsItem(Model):
    """Series of calculated metrics for a specific set of repositories and developers."""

    def __init__(self,
                 for_: Optional[ForSet] = None,
                 values: Optional[List[CalculatedPullRequestMetricValues]] = None):
        """CalculatedPullRequestMetricsItem - a model defined in OpenAPI

        :param for_: The for_ of this CalculatedPullRequestMetricsItem.
        :param values: The values of this CalculatedPullRequestMetricsItem.
        """
        self.openapi_types = {"for_": ForSet, "values": List[CalculatedPullRequestMetricValues]}

        self.attribute_map = {"for_": "for", "values": "values"}

        self._for_ = for_
        self._values = values

    @property
    def for_(self):
        """Gets the for_ of this CalculatedPullRequestMetricsItem.

        :return: The for of this CalculatedPullRequestMetricsItem.
        :rtype: ForSet
        """
        return self._for_

    @for_.setter
    def for_(self, for_):
        """Sets the for_ of this CalculatedPullRequestMetricsItem.

        :param for_: The for of this CalculatedPullRequestMetricsItem.
        :type for_: ForSet
        """
        if for_ is None:
            raise ValueError("Invalid value for `for`, must not be `None`")

        self._for_ = for_

    @property
    def values(self):
        """Gets the values of this CalculatedPullRequestMetricsItem.

        :return: The values of this CalculatedPullRequestMetricsItem.
        :rtype: List[CalculatedPullRequestMetricValues]
        """
        return self._values

    @values.setter
    def values(self, values):
        """Sets the values of this CalculatedPullRequestMetricsItem.

        :param values: The values of this CalculatedPullRequestMetricsItem.
        :type values: List[CalculatedPullRequestMetricValues]
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values
