from enum import Enum
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_issue import JIRAIssue


class GetJIRAIssuesInclude(Enum):
    """Additional information to include in the response for the get_jira_issues operation."""

    JIRA_USERS = "jira_users"


class GetJIRAIssuesRequest(Model):
    """Request body for the search_jira_issues operation."""

    account: int
    issues: list[str]
    include: Optional[list[str]]  # values from GetJIRAIssuesInclude


class GetJIRAIssuesResponse(Model):
    """Response for the search_jira_issues operation."""

    issues: list[JIRAIssue]
