from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import \
    CalculatedLinearMetricValues
from athenian.api.models.web.for_set_pull_requests import ForSet
from athenian.api.models.web.granularity import GranularityMixin


class CalculatedPullRequestMetricsItem(Model, GranularityMixin):
    """Series of calculated metrics for a specific set of repositories and developers."""

    openapi_types = {
        "for_": ForSet,
        "granularity": str,
        "values": List[CalculatedLinearMetricValues],
    }

    attribute_map = {"for_": "for", "granularity": "granularity", "values": "values"}

    def __init__(self,
                 for_: Optional[ForSet] = None,
                 granularity: Optional[str] = None,
                 values: Optional[List[CalculatedLinearMetricValues]] = None):
        """CalculatedPullRequestMetricsItem - a model defined in OpenAPI

        :param for_: The for_ of this CalculatedPullRequestMetricsItem.
        :param granularity: The granularity of this CalculatedPullRequestMetricsItem.
        :param values: The values of this CalculatedPullRequestMetricsItem.
        """
        self._for_ = for_
        self._granularity = granularity
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
        :rtype: List[CalculatedLinearMetricValues]
        """
        return self._values

    @values.setter
    def values(self, values):
        """Sets the values of this CalculatedPullRequestMetricsItem.

        :param values: The values of this CalculatedPullRequestMetricsItem.
        :type values: List[CalculatedLinearMetricValues]
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values
