from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.code_check_histogram_definition import CodeCheckHistogramDefinition
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import QuantilesMixin
from athenian.api.models.web.for_set_code_checks import ForSetCodeChecks


class _CodeCheckHistogramsRequest(Model, QuantilesMixin, sealed=False):
    """Request of `/histograms/code_checks`."""

    for_: (list[ForSetCodeChecks], "for")
    histograms: list[CodeCheckHistogramDefinition]
    quantiles: list[float]
    split_by_check_runs: Optional[bool]


CodeCheckHistogramsRequest = AllOf(
    _CodeCheckHistogramsRequest,
    CommonFilterProperties,
    name="CodeCheckHistogramsRequest",
    module=__name__,
)
