from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.deployment_analysis_code import DeploymentAnalysisCode
from athenian.api.models.web.filtered_release import FilteredRelease


class DeploymentAnalysisUnsealed(Model, sealed=False):
    """Statistics and contents of the deployment."""

    code: DeploymentAnalysisCode
    releases: Optional[List[FilteredRelease]]


class DeploymentAnalysis(DeploymentAnalysisUnsealed):
    """Result the analysis of the deployment."""
