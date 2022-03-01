from typing import Dict, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.deployment_notification import DeploymentNotification


class _IncludedDeployments(Model, sealed=False):
    """Mentioned deployments."""

    openapi_types = {"deployments": Optional[Dict[str, DeploymentNotification]]}
    attribute_map = {"deployments": "deployments"}

    def __init__(self, deployments: Optional[Dict[str, DeploymentNotification]] = None):
        """IncludedDeployments - a model defined in OpenAPI

        :param deployments: The deployments of this IncludedDeployments.
        """
        self._deployments = deployments

    @property
    def deployments(self) -> Optional[Dict[str, DeploymentNotification]]:
        """Gets the deployments of this IncludedDeployments.

        Mapping deployment names to their details.

        :return: The deployments of this IncludedDeployments.
        """
        return self._deployments

    @deployments.setter
    def deployments(self, deployments: Optional[Dict[str, DeploymentNotification]]):
        """Sets the deployments of this IncludedDeployments.

        Mapping deployment names to their details.

        :param deployments: The deployments of this IncludedDeployments.
        """
        self._deployments = deployments


class IncludedDeployments(_IncludedDeployments):
    """Mentioned deployments."""
