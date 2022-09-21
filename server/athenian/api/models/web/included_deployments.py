from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.deployment_notification import DeploymentNotification


class _IncludedDeployments(Model, sealed=False):
    """Mentioned deployments."""

    deployments: Optional[dict[str, DeploymentNotification]]


class IncludedDeployments(_IncludedDeployments):
    """Mentioned deployments."""
