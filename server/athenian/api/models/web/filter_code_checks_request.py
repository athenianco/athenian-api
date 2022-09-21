from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import QuantilesMixin
from athenian.api.models.web.for_set_common import CommonPullRequestFilters


class _FilterCodeChecksRequest(Model, QuantilesMixin, sealed=False):
    """Request body of `/filter/code_checks`."""

    in_: (list[str], "in")
    triggered_by: Optional[list[str]]
    quantiles: Optional[list[float]]


FilterCodeChecksRequest = AllOf(
    _FilterCodeChecksRequest,
    CommonFilterProperties,
    CommonPullRequestFilters,
    name="FilterCodeChecksRequest",
    module=__name__,
)
