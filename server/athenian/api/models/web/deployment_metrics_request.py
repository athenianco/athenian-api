from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import CommonMetricsProperties
from athenian.api.models.web.deployment_metric_id import DeploymentMetricID
from athenian.api.models.web.for_set_deployments import ForSetDeployments


class _DeploymentMetricsRequest(Model):
    """Request body of `/metrics/deployments`, the deployments selector."""

    attribute_types = {
        "for_": List[ForSetDeployments],
        "metrics": List[str],
    }

    attribute_map = {
        "for_": "for",
        "metrics": "metrics",
    }

    def __init__(
        self,
        for_: Optional[List[ForSetDeployments]] = None,
        metrics: Optional[List[str]] = None,
    ):
        """DeploymentMetricsRequest - a model defined in OpenAPI

        :param for_: The for_ of this DeploymentMetricsRequest.
        :param metrics: The metrics of this DeploymentMetricsRequest.
        """
        self._for_ = for_
        self._metrics = metrics

    @property
    def for_(self) -> List[ForSetDeployments]:
        """Gets the for_ of this DeploymentMetricsRequest.

        List of repository groups for which to calculate the metrics.

        :return: The for_ of this DeploymentMetricsRequest.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: List[ForSetDeployments]):
        """Sets the for_ of this DeploymentMetricsRequest.

        List of repository groups for which to calculate the metrics.

        :param for_: The for_ of this DeploymentMetricsRequest.
        """
        self._for_ = for_

    @property
    def metrics(self) -> List[str]:
        """Gets the metrics of this DeploymentMetricsRequest.

        Requested metric identifiers.

        :return: The metrics of this DeploymentMetricsRequest.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[str]):
        """Sets the metrics of this DeploymentMetricsRequest.

        Requested metric identifiers.

        :param metrics: The metrics of this DeploymentMetricsRequest.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for metric in metrics:
            if metric not in DeploymentMetricID:
                raise ValueError(f"Deployment metric {metric} is unsupported.")

        self._metrics = metrics


DeploymentMetricsRequest = AllOf(_DeploymentMetricsRequest,
                                 CommonFilterProperties,
                                 CommonMetricsProperties,
                                 name="DeploymentMetricsRequest",
                                 module=__name__)
