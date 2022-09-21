from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_metrics_properties import QuantilesMixin
from athenian.api.models.web.filter_jira_common import FilterJIRACommon
from athenian.api.models.web.jira_filter_with import JIRAFilterWith
from athenian.api.models.web.jira_histogram_definition import JIRAHistogramDefinition


class _JIRAHistogramsRequest(Model, QuantilesMixin, sealed=False):
    """Request of `/histograms/jira`."""

    epics: Optional[List[str]]
    with_: (Optional[List[JIRAFilterWith]], "with")
    histograms: List[JIRAHistogramDefinition]
    quantiles: Optional[List[float]]

    def validate_histograms(
        self,
        histograms: list[JIRAHistogramDefinition],
    ) -> list[JIRAHistogramDefinition]:
        """Sets the histograms of this JIRAHistogramsRequest.

        Histogram parameters for each wanted topic.

        :param histograms: The histograms of this JIRAHistogramsRequest.
        """
        if histograms is None:
            raise ValueError("Invalid value for `histograms`, must not be `None`")
        return histograms


JIRAHistogramsRequest = AllOf(
    _JIRAHistogramsRequest, FilterJIRACommon, name="JIRAHistogramsRequest", module=__name__,
)
