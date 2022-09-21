from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.typing_utils import VerbatimOptional


class ReleasedPullRequest(Model):
    """Details about a pull request listed in `/filter/releases`."""

    number: int
    title: str
    additions: int
    deletions: int
    author: VerbatimOptional[str]
    jira: Optional[list[str]]
