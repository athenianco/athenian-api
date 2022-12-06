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


class SearchPullRequestsOrderByStageTiming(Enum):
    """Stage timing that can be used as orderby fields when searching for pull requests."""

    PR_WIP_STAGE_TIMING = "pr-wip-stage-timing"
    PR_REVIEW_STAGE_TIMING = "pr-review-stage-timing"
    PR_MERGE_STAGE_TIMING = "pr-merge-stage-timing"
    PR_RELEASE_STAGE_TIMING = "pr-release-stage-timing"


class SearchPullRequestsOrderByPRTrait(Enum):
    """A trait of the pull request usable as ordering fine in /search/pull_requests."""

    WORK_BEGAN = "pr-order-by-work-began"
    FIRST_REVIEW_REQUEST = "pr-order-by-first-review-request"


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
    stages: Optional[list[str]]


class PullRequestDigest(Model):
    """Basic information about a pull request."""

    number: int
    repository: str


class SearchPullRequestsResponse(Model):
    """List of pull requests result of the search operation."""

    pull_requests: Sequence[PullRequestDigest]
