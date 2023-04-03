from typing import Any

from athenian.api.models.web.base_model_ import Model


class DeploymentLabelsResponse(Model):
    """Container for the labels associated with the deployments."""

    labels: dict[str, Any]
