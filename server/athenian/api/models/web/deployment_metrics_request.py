from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import CommonMetricsProperties
from athenian.api.models.web.deployment_metric_id import DeploymentMetricID
from athenian.api.models.web.for_set_deployments import ForSetDeployments


class _DeploymentMetricsRequest(Model):
    """Request body of `/metrics/deployments`, the deployments selector."""

    for_: (list[ForSetDeployments], "for")
    metrics: list[str]

    def validate_metrics(self, metrics: list[str]) -> list[str]:
        """Sets the metrics of this DeploymentMetricsRequest.

        Requested metric identifiers.

        :param metrics: The metrics of this DeploymentMetricsRequest.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for metric in metrics:
            if metric not in DeploymentMetricID:
                raise ValueError(f"Deployment metric {metric} is unsupported.")

        return metrics


DeploymentMetricsRequest = AllOf(
    _DeploymentMetricsRequest,
    CommonFilterProperties,
    CommonMetricsProperties,
    name="DeploymentMetricsRequest",
    module=__name__,
)
