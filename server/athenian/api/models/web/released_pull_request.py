from datetime import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.typing_utils import VerbatimOptional


class _PullRequestSummary(Model, sealed=False):
    """Common information about a pull request."""

    number: int
    title: str
    created: datetime
    additions: int
    deletions: int
    author: VerbatimOptional[str]
    jira: Optional[list[str]]


class ReleasedPullRequest(_PullRequestSummary):
    """Details about a pull request listed in `/filter/releases`."""
