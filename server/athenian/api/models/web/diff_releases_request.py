from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.release_pair import ReleasePair


class DiffReleasesRequest(Model):
    """Request of `/diff/releases`. Define pairs of releases for several repositories to find \
    the releases in between."""

    account: int
    borders: dict[str, list[ReleasePair]]
