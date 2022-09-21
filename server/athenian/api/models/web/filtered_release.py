from datetime import datetime, timedelta
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.released_pull_request import ReleasedPullRequest
from athenian.api.typing_utils import VerbatimOptional


class FilteredRelease(Model):
    """Various information about a repository release."""

    name: str
    sha: str
    repository: str
    url: str
    published: datetime
    age: timedelta
    added_lines: int
    deleted_lines: int
    commits: int
    publisher: VerbatimOptional[str]
    commit_authors: list[str]
    prs: list[ReleasedPullRequest]
    deployments: Optional[list[str]]
