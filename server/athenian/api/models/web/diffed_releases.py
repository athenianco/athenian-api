from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.release_diff import ReleaseDiff
from athenian.api.models.web.release_set import ReleaseSetInclude


class DiffedReleases(Model):
    """Response of `/diff/releases` - the found inner releases for each repository."""

    include: ReleaseSetInclude
    data: dict[str, list[ReleaseDiff]]
