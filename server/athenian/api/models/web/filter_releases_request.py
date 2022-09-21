from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.for_set_common import CommonPullRequestFilters
from athenian.api.models.web.release_with import ReleaseWith


class _FilterReleasesRequest(Model, sealed=False):
    """Structure to specify the filter traits of releases."""

    in_: (List[str], "in")
    with_: (Optional[ReleaseWith], "with")


FilterReleasesRequest = AllOf(
    _FilterReleasesRequest,
    CommonFilterProperties,
    CommonPullRequestFilters,
    name="FilterReleasesRequest",
    module=__name__,
)
