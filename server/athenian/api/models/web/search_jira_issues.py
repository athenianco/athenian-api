from datetime import date
from enum import Enum
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_filter import JIRAFilter
from athenian.api.models.web.jira_filter_with import JIRAFilterWith
from athenian.api.models.web.search_prs import OrderByExpression


class SearchJIRAIssuesOrderByExpression(OrderByExpression):
    """An expression based on a field used to sort jira issues."""


class SearchJIRAIssuesRequest(Model):
    """Query conditions to search for Jira issues."""

    account: int
    date_from: Optional[date]
    date_to: Optional[date]
    filter: Optional[JIRAFilter]
    with_: Optional[JIRAFilterWith]
    order_by: Optional[list[SearchJIRAIssuesOrderByExpression]]

    attribute_map = {
        "with_": "with",
    }


class JIRAIssueDigest(Model):
    """Basic information about a Jira issue."""

    id: str


class SearchJIRAIssuesResponse(Model):
    """Result of the search for Jira issues."""

    jira_issues: list[JIRAIssueDigest]


class SearchJIRAIssuesOrderByIssueTrait(Enum):
    """A trait of the Jira issue usable as ordering field in /search/jira_issues."""

    CREATED = "created"
    UPDATED = "updated"
    WORK_BEGAN = "work_began"
