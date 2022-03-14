from athenian.api.models.web.base_model_ import Enum, Model


class JIRAMetricID(Model, metaclass=Enum):
    """Currently supported JIRA activity metrics."""

    JIRA_OPEN = "jira-open"
    JIRA_RESOLVED = "jira-resolved"
    JIRA_RAISED = "jira-raised"
    JIRA_ACKNOWLEDGED = "jira-acknowledged"
    JIRA_ACKNOWLEDGED_Q = "jira-acknowledged-q"
    JIRA_LIFE_TIME = "jira-life-time"
    JIRA_LEAD_TIME = "jira-lead-time"
    JIRA_ACKNOWLEDGE_TIME = "jira-acknowledge-time"
    JIRA_RESOLUTION_RATE = "jira-resolution-rate"
    JIRA_PR_LAG_TIME = "jira-pr-lag-time"
    JIRA_BACKLOG_TIME = "jira-backlog-time"
