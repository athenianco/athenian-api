from typing import Optional

from athenian.api.models.web.base_model_ import Model


class LogicalDeploymentRules(Model):
    """Rules to match deployments to logical repository."""

    title: Optional[str]
    labels_include: Optional[dict[str, list[str]]]
