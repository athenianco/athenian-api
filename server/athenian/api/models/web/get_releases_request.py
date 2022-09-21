from typing import List

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.release_names import ReleaseNames


class GetReleasesRequest(Model):
    """Request body of `/get/releases`. Declaration of which releases the user wants to list."""

    account: int
    releases: List[ReleaseNames]
