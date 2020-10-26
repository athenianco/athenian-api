from athenian.api.models.web.base_model_ import Enum, Model


class JIRAMetricID(Model, metaclass=Enum):
    """Currently supported JIRA activity metrics."""

    JIRA_BUG_OPEN = "jira-bug-open"
    JIRA_BUG_RESOLVED = "jira-bug-resolved"
    JIRA_BUG_RAISED = "jira-bug-raised"
    JIRA_MTT_RESTORE = "jira-mtt-restore"
    JIRA_MTT_REPAIR = "jira-mtt-repair"
