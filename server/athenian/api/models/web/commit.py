from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.commit_signature import CommitSignature


class Commit(Model):
    """Information about a commit."""

    repository: str
    hash: str
    children: Optional[list[str]]
    deployments: Optional[list[str]]
    author: CommitSignature
    committer: CommitSignature
    message: str
    size_added: int
    size_removed: int
    files_changed: int
