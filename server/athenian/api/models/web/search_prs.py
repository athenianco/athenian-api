from datetime import date
from enum import Enum
from typing import Optional, Sequence

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_filter import JIRAFilter
from athenian.api.models.web.pull_request_with import PullRequestWith


class OrderByDirection(Enum):
    """Direction of an order by."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


class SearchPullRequestsOrderByExpression(Model):
    """An expression based on a field used to sort pull requests."""

    field: str
    exclude_nulls: bool = True
    direction: str = OrderByDirection.ASCENDING.value
    nulls_first: bool = False


class SearchPullRequestsRequest(Model):
    """Query condition to search for pull requests."""

    account: int
    date_from: date
    date_to: date
    jira: Optional[JIRAFilter]
    order_by: Optional[list[SearchPullRequestsOrderByExpression]]
    repositories: Optional[list[str]]
    participants: Optional[PullRequestWith]


class PullRequestDigest(Model):
    """Basic information about a pull request."""

    number: int
    repository: str


class SearchPullRequestsResponse(Model):
    """List of pull requests result of the search operation."""

    pull_requests: Sequence[PullRequestDigest]
