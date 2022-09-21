from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.linked_jira_issue import LinkedJIRAIssue


class _IncludedJIRAIssues(Model, sealed=False):
    """Mentioned JIRA issues."""

    jira: Optional[dict[str, LinkedJIRAIssue]]


class IncludedJIRAIssues(_IncludedJIRAIssues):
    """Mentioned JIRA issues."""
