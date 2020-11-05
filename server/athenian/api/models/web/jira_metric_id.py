from athenian.api.models.web.base_model_ import Enum, Model


class JIRAMetricID(Model, metaclass=Enum):
    """Currently supported JIRA activity metrics."""

    JIRA_OPEN = "jira-open"
    JIRA_RESOLVED = "jira-resolved"
    JIRA_RAISED = "jira-raised"
    JIRA_LIFE_TIME = "jira-life-time"
    JIRA_LEAD_TIME = "jira-lead-time"
    JIRA_FLOW_RATIO = "jira-flow-ratio"
