from datetime import date
from typing import Optional, Sequence

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_filter import JIRAFilter
from athenian.api.models.web.pull_request_with import PullRequestWith


class SearchPullRequestsRequest(Model):
    """Query condition to search for pull requests."""

    account: int
    date_from: date
    date_to: date
    jira: Optional[JIRAFilter]
    repositories: Optional[list[str]]
    participants: Optional[PullRequestWith]


class PullRequestDigest(Model):
    """Basic information about a pull request."""

    number: int
    repository: str


class SearchPullRequestsResponse(Model):
    """List of pull requests result of the search operation."""

    pull_requests: Sequence[PullRequestDigest]
