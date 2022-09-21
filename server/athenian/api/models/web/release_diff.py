from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.filtered_release import FilteredRelease


class ReleaseDiff(Model):
    """Inner releases between `old` and `new`, including the latter."""

    old: str
    new: str
    releases: list[FilteredRelease]
