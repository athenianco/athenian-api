from athenian.api.models.web.base_model_ import Enum, Model


class JIRAMetricID(Model, metaclass=Enum):
    """Currently supported JIRA activity metrics."""

    JIRA_OPEN = "jira-open"
    JIRA_RESOLVED = "jira-resolved"
    JIRA_RAISED = "jira-raised"
    JIRA_MTT_RESTORE = "jira-mtt-restore"
    JIRA_MTT_REPAIR = "jira-mtt-repair"
