from athenian.api.models.web.base_model_ import Enum, Model


class JIRAMetricID(Model, metaclass=Enum):
    """Currently supported JIRA activity metrics."""

    BUG_OPEN = "jira-bug-open"
    BUG_RESOLVED = "jira-bug-resolved"
    BUG_RAISED = "jira-bug-raised"
