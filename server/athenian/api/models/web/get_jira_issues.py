from enum import Enum
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.included_native_user import IncludedNativeUser
from athenian.api.models.web.jira_issue import JIRAIssue
from athenian.api.models.web.jira_user import JIRAUser


class GetJIRAIssuesInclude(Enum):
    """Additional information to include in the response for the get_jira_issues operation."""

    GITHUB_USERS = "github_users"
    JIRA_USERS = "jira_users"


class GetJIRAIssuesRequest(Model):
    """Request body for the search_jira_issues operation."""

    account: int
    issues: list[str]
    include: Optional[list[str]]  # values from GetJIRAIssuesInclude


class GetJIRAIssuesResponseInclude(Model):
    """Additional information included in the get_jira_issues response."""

    github_users: Optional[dict[str, IncludedNativeUser]]
    jira_users: Optional[list[JIRAUser]]


class GetJIRAIssuesResponse(Model):
    """Response for the search_jira_issues operation."""

    issues: list[JIRAIssue]
    include: Optional[GetJIRAIssuesResponseInclude]
