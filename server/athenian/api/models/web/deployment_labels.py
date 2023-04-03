from typing import Any

from athenian.api.models.web.base_model_ import Model

_DeploymentLabels = dict[str, Any]


class DeploymentLabelsResponse(Model):
    """Container for the labels associated with the deployments."""

    labels: _DeploymentLabels


class DeploymentModifyLabelsRequest(Model):
    """Request to modify the labels associated with the deployment."""

    delete: list[str] | None
    upsert: _DeploymentLabels | None
