from typing import Dict, List, Optional

from athenian.api.models.web.base_model_ import Model


class DeploymentAnalysisCode(Model):
    """Summary of the deployed code."""

    prs: Dict[str, int]
    lines_prs: Dict[str, int]
    lines_overall: Dict[str, int]
    commits_prs: Dict[str, int]
    commits_overall: Dict[str, int]
    jira: Optional[Dict[str, List[str]]]
