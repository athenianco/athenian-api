from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_deployment_properties import CommonDeploymentProperties
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import CommonMetricsProperties
from athenian.api.models.web.deployment_metric_id import DeploymentMetricID
from athenian.api.models.web.deployment_with import DeploymentWith
from athenian.api.models.web.for_set import make_common_pull_request_filters


class _DeploymentMetricsRequest(Model):
    """Request body of `/metrics/deployments`, the deployments selector."""

    openapi_types = {
        "for_": Optional[List[List[str]]],
        "with_": Optional[List[DeploymentWith]],
        "metrics": List[str],
        "environments": Optional[List[List[str]]],
    }

    attribute_map = {
        "for_": "for",
        "with_": "with",
        "metrics": "metrics",
        "environments": "environments",
    }

    def __init__(
        self,
        for_: Optional[List[List[str]]] = None,
        with_: Optional[List[DeploymentWith]] = None,
        metrics: Optional[List[str]] = None,
        environments: Optional[List[List[str]]] = None,
    ):
        """DeploymentMetricsRequest - a model defined in OpenAPI

        :param for_: The for_ of this DeploymentMetricsRequest.
        :param with_: The with_ of this DeploymentMetricsRequest.
        :param metrics: The metrics of this DeploymentMetricsRequest.
        :param environments: The environments of this DeploymentMetricsRequest.
        """
        self._for_ = for_
        self._with_ = with_
        self._metrics = metrics
        self._environments = environments

    @property
    def for_(self) -> Optional[List[List[str]]]:
        """Gets the for_ of this DeploymentMetricsRequest.

        List of repository groups for which to calculate the metrics.

        :return: The for_ of this DeploymentMetricsRequest.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: Optional[List[List[str]]]):
        """Sets the for_ of this DeploymentMetricsRequest.

        List of repository groups for which to calculate the metrics.

        :param for_: The for_ of this DeploymentMetricsRequest.
        """
        self._for_ = for_

    @property
    def with_(self) -> Optional[List[DeploymentWith]]:
        """Gets the with_ of this DeploymentMetricsRequest.

        List of developer groups for which to calculate the metrics.

        :return: The with_ of this DeploymentMetricsRequest.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[List[DeploymentWith]]):
        """Sets the with_ of this DeploymentMetricsRequest.

        List of developer groups for which to calculate the metrics.

        :param with_: The with_ of this DeploymentMetricsRequest.
        """
        self._with_ = with_

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

    @property
    def environments(self) -> Optional[List[List[str]]]:
        """Gets the environments of this DeploymentMetricsRequest.

        List of environment groups for which to calculate the metrics.

        :return: The environments of this DeploymentMetricsRequest.
        """
        return self._environments

    @environments.setter
    def environments(self, environments: Optional[List[List[str]]]):
        """Sets the environments of this DeploymentMetricsRequest.

        List of environment groups for which to calculate the metrics.

        :param environments: The environments of this DeploymentMetricsRequest.
        """
        self._environments = environments


DeploymentMetricsRequest = AllOf(_DeploymentMetricsRequest,
                                 CommonFilterProperties,
                                 CommonMetricsProperties,
                                 make_common_pull_request_filters("pr_"),
                                 CommonDeploymentProperties,
                                 name="DeploymentMetricsRequest",
                                 module=__name__)
