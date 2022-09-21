from typing import Optional

from athenian.api.models.web.base_model_ import Model


class _PullRequestLabel(Model, sealed=False):
    name: str
    description: Optional[str]
    color: str


class PullRequestLabel(_PullRequestLabel):
    """Pull request label."""
