from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.deployment_notification import DeploymentNotification
from athenian.api.models.web.jira_epic import JIRAEpic
from athenian.api.models.web.jira_issue import JIRAIssue
from athenian.api.models.web.jira_issue_type import JIRAIssueType
from athenian.api.models.web.jira_label import JIRALabel
from athenian.api.models.web.jira_priority import JIRAPriority
from athenian.api.models.web.jira_status import JIRAStatus
from athenian.api.models.web.jira_user import JIRAUser


class FilteredJIRAStuff(Model):
    """Response from `/filter/jira`: found JIRA epics, labels, issue types, priorities, and \
    mentioned users."""

    epics: list[JIRAEpic]
    issues: list[JIRAIssue]
    labels: list[JIRALabel]
    issue_types: list[JIRAIssueType]
    priorities: list[JIRAPriority]
    statuses: list[JIRAStatus]
    users: list[JIRAUser]
    deployments: Optional[dict[str, DeploymentNotification]]
