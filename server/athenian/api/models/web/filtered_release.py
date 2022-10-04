from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_release import CommonRelease
from athenian.api.models.web.released_pull_request import ReleasedPullRequest


class _FilteredRelease(Model):
    """Specific information about a repository release."""

    prs: list[ReleasedPullRequest]
    deployments: Optional[list[str]]


FilteredRelease = AllOf(CommonRelease, _FilteredRelease, name="FilteredRelease", module=__name__)
