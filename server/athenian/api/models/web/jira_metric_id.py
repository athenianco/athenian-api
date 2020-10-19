from athenian.api.models.web.base_model_ import Enum


class JIRAMetricID(Enum):
    """Currently supported JIRA activity metrics."""

    OPEN = "jira-bug-open"
    RESOLVED = "jira-bug-resolved"
    RAISED = "jira-bug-raised"
