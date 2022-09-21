from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties


class _FilterRepositoriesRequest(Model, sealed=False):
    """Structure to specify the filter traits of repositories."""

    in_: (Optional[list[str]], "in")
    exclude_inactive: Optional[bool]


FilterRepositoriesRequest = AllOf(
    _FilterRepositoriesRequest,
    CommonFilterProperties,
    name="FilterRepositoriesRequest",
    module=__name__,
)
