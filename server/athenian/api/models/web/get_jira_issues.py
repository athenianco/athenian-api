from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_issue import JIRAIssue


class GetJIRAIssuesRequest(Model):
    """Request body for the search_jira_issues operation."""

    account: int
    issues: list[str]


class GetJIRAIssuesResponse(Model):
    """Response for the search_jira_issues operation."""

    issues: list[JIRAIssue]
