from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class PullRequestWith(Model):
    """Triage PRs by various developer participation."""

    author: Optional[List[str]]
    reviewer: Optional[List[str]]
    commit_author: Optional[List[str]]
    commit_committer: Optional[List[str]]
    commenter: Optional[List[str]]
    merger: Optional[List[str]]
    releaser: Optional[List[str]]
