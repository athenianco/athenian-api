from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import \
    CalculatedLinearMetricValues
from athenian.api.models.web.granularity import GranularityMixin
from athenian.api.models.web.jira_filter_with import JIRAFilterWith


class CalculatedJIRAMetricValues(Model, GranularityMixin):
    """Calculated JIRA metrics for a specific granularity."""

    attribute_types = {
        "granularity": str,
        "jira_label": Optional[str],
        "with_": Optional[JIRAFilterWith],
        "values": List[CalculatedLinearMetricValues],
    }

    attribute_map = {
        "granularity": "granularity",
        "jira_label": "jira_label",
        "with_": "with",
        "values": "values",
    }

    def __init__(self,
                 granularity: Optional[str] = None,
                 jira_label: Optional[str] = None,
                 with_: Optional[JIRAFilterWith] = None,
                 values: Optional[List[CalculatedLinearMetricValues]] = None):
        """CalculatedJIRAMetricValues - a model defined in OpenAPI

        :param granularity: The granularity of this CalculatedJIRAMetricValues.
        :param jira_label: The jira_label of this CalculatedJIRAMetricValues.
        :param with_: The with of this CalculatedJIRAMetricValues.
        :param values: The values of this CalculatedJIRAMetricValues.
        """
        self._granularity = granularity
        self._jira_label = jira_label
        self._with_ = with_
        self._values = values

    @property
    def jira_label(self) -> Optional[str]:
        """Gets the jira_label of this CalculatedJIRAMetricValues.

        Title of the assigned JIRA label, if `group_by_jira_label` was previously set to `true`.

        :return: The jira_label of this CalculatedJIRAMetricValues.
        """
        return self._jira_label

    @jira_label.setter
    def jira_label(self, jira_label: Optional[str]):
        """Sets the jira_label of this CalculatedJIRAMetricValues.

        Title of the assigned JIRA label, if `group_by_jira_label` was previously set to `true`.

        :param jira_label: The jira_label of this CalculatedJIRAMetricValues.
        """
        self._jira_label = jira_label

    @property
    def with_(self) -> Optional[List[JIRAFilterWith]]:
        """Gets the with of this CalculatedJIRAMetricValues.

        Groups of issue participants. The metrics will be calculated for each group.

        :return: The with of this CalculatedJIRAMetricValues.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[List[JIRAFilterWith]]):
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
