from typing import List

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import CalculatedLinearMetricValues
from athenian.api.models.web.deployment_metric_id import DeploymentMetricID
from athenian.api.models.web.for_set_deployments import ForSetDeployments


class CalculatedDeploymentMetric(Model):
    """Calculated metrics for a deployments group."""

    for_: (ForSetDeployments, "for")
    metrics: List[DeploymentMetricID]
    granularity: str
    values: List[CalculatedLinearMetricValues]

    def validate_metrics(self, metrics: List[str]):
        """Sets the metrics of this CalculatedDeploymentMetric.

        :param metrics: The metrics of this CalculatedDeploymentMetric.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for metric in metrics:
            if metric not in DeploymentMetricID:
                raise ValueError(f"Deployment metric {metric} is unsupported.")

        return metrics
