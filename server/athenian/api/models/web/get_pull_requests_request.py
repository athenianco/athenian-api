from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.pull_request_numbers import PullRequestNumbers


class GetPullRequestsRequest(Model):
    """Request body of `/get/pull_requests`. Declaration of which PRs the user wants to analyze."""

    account: int
    prs: List[PullRequestNumbers]
    environments: Optional[list[str]]
