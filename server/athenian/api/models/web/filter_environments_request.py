from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties


class _FilterEnvironmentsRequest(Model, sealed=False):
    """Request body of `/filter/environments`. Filters for deployment environments."""

    repositories: Optional[list[str]]


FilterEnvironmentsRequest = AllOf(
    _FilterEnvironmentsRequest,
    CommonFilterProperties,
    name="FilterEnvironmentsRequest",
    module=__name__,
)
