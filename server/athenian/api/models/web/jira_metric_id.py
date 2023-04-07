from athenian.api.models.web.base_model_ import Enum, Model


class JIRAMetricID(Model, metaclass=Enum):
    """Currently supported JIRA activity metrics."""

    JIRA_OPEN = "jira-open"
    JIRA_RESOLVED = "jira-resolved"
    JIRA_RAISED = "jira-raised"
    JIRA_ACKNOWLEDGED = "jira-acknowledged"
    JIRA_ACKNOWLEDGED_Q = "jira-acknowledged-q"
    JIRA_LIFE_TIME = "jira-life-time"
    JIRA_LIFE_TIME_BELOW_THRESHOLD_RATIO = "jira-life-time-below-threshold-ratio"
    JIRA_LEAD_TIME = "jira-lead-time"
    JIRA_LEAD_TIME_BELOW_THRESHOLD_RATIO = "jira-lead-time-below-threshold-ratio"
    JIRA_ACKNOWLEDGE_TIME = "jira-acknowledge-time"
    JIRA_ACKNOWLEDGE_TIME_BELOW_THRESHOLD_RATIO = "jira-acknowledge-time-below-threshold-ratio"
    JIRA_RESOLUTION_RATE = "jira-resolution-rate"
    JIRA_PR_LAG_TIME = "jira-pr-lag-time"
    JIRA_BACKLOG_TIME = "jira-backlog-time"
