from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.deployed_release import DeployedRelease
from athenian.api.models.web.deployment_analysis_code import DeploymentAnalysisCode
from athenian.api.models.web.released_pull_request import ReleasedPullRequest


class DeploymentAnalysisUnsealed(Model, sealed=False):
    """Statistics and contents of the deployment."""

    code: DeploymentAnalysisCode
    prs: list[ReleasedPullRequest]
    releases: Optional[list[DeployedRelease]]


class DeploymentAnalysis(DeploymentAnalysisUnsealed):
    """Result the analysis of the deployment."""
