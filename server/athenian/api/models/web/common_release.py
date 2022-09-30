from datetime import datetime, timedelta

from athenian.api.models.web.base_model_ import Model
from athenian.api.typing_utils import VerbatimOptional


class CommonRelease(Model, sealed=False):
    """Various information about a repository release, shared fields."""

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
