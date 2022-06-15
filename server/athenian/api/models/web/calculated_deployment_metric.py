from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import CalculatedLinearMetricValues
from athenian.api.models.web.deployment_metric_id import DeploymentMetricID
from athenian.api.models.web.for_set_deployments import ForSetDeployments


class CalculatedDeploymentMetric(Model):
    """Calculated metrics for a deployments group."""

    attribute_types = {
        "for_": ForSetDeployments,
        "metrics": List[DeploymentMetricID],
        "granularity": str,
        "values": List[CalculatedLinearMetricValues],
    }

    attribute_map = {
        "for_": "for",
        "metrics": "metrics",
        "granularity": "granularity",
        "values": "values",
    }

    def __init__(
        self,
        for_: Optional[ForSetDeployments] = None,
        metrics: Optional[List[str]] = None,
        granularity: Optional[str] = None,
        values: Optional[List[CalculatedLinearMetricValues]] = None,
    ):
        """CalculatedDeploymentMetric - a model defined in OpenAPI

        :param for_: The for_ of this CalculatedDeploymentMetric.
        :param metrics: The metrics of this CalculatedDeploymentMetric.
        :param granularity: The granularity of this CalculatedDeploymentMetric.
        :param values: The values of this CalculatedDeploymentMetric.
        """
        self._for_ = for_
        self._metrics = metrics
        self._granularity = granularity
        self._values = values

    @property
    def for_(self) -> ForSetDeployments:
        """Gets the for_ of this CalculatedDeploymentMetric.

        :return: The for_ of this CalculatedDeploymentMetric.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: ForSetDeployments):
        """Sets the for_ of this CalculatedDeploymentMetric.

        :param for_: The for_ of this CalculatedDeploymentMetric.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for`, must not be `None`")

        self._for_ = for_

    @property
    def metrics(self) -> List[str]:
        """Gets the metrics of this CalculatedDeploymentMetric.

        :return: The metrics of this CalculatedDeploymentMetric.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[str]):
        """Sets the metrics of this CalculatedDeploymentMetric.

        :param metrics: The metrics of this CalculatedDeploymentMetric.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for metric in metrics:
            if metric not in DeploymentMetricID:
                raise ValueError(f"Deployment metric {metric} is unsupported.")

        self._metrics = metrics

    @property
    def granularity(self) -> str:
        """Gets the granularity of this CalculatedDeploymentMetric.

        How often the metrics are reported. The value must satisfy the following regular
        expression: /^all|(([1-9]\\d* )?(aligned )?(day|week|month|year))$/. \"all\" produces
        a single interval [`date_from`, `date_to`]. \"aligned week/month/year\" produces
        intervals cut by calendar week/month/year borders, for example, when `date_from` is
        `2020-01-15` and `date_to` is `2020-03-10`, the intervals will be
        `2020-01-15` - `2020-02-01` - `2020-03-01` - `2020-03-10`.

        :return: The granularity of this CalculatedDeploymentMetric.
        """
        return self._granularity

    @granularity.setter
    def granularity(self, granularity: str):
        """Sets the granularity of this CalculatedDeploymentMetric.

        How often the metrics are reported. The value must satisfy the following regular
        expression: /^all|(([1-9]\\d* )?(aligned )?(day|week|month|year))$/. \"all\" produces
        a single interval [`date_from`, `date_to`]. \"aligned week/month/year\" produces
        intervals cut by calendar week/month/year borders, for example, when `date_from` is
        `2020-01-15` and `date_to` is `2020-03-10`, the intervals will be
        `2020-01-15` - `2020-02-01` - `2020-03-01` - `2020-03-10`.

        :param granularity: The granularity of this CalculatedDeploymentMetric.
        """
        if granularity is None:
            raise ValueError("Invalid value for `granularity`, must not be `None`")

        self._granularity = granularity

    @property
    def values(self) -> List[CalculatedLinearMetricValues]:
        """Gets the values of this CalculatedDeploymentMetric.

        The sequence steps from `date_from` till `date_to` by `granularity`.

        :return: The values of this CalculatedDeploymentMetric.
        """
        return self._values

    @values.setter
    def values(self, values: List[CalculatedLinearMetricValues]):
        """Sets the values of this CalculatedDeploymentMetric.

        The sequence steps from `date_from` till `date_to` by `granularity`.

        :param values: The values of this CalculatedDeploymentMetric.
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values
