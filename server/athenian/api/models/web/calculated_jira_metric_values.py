from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import CalculatedLinearMetricValues
from athenian.api.models.web.granularity import GranularityMixin
from athenian.api.models.web.jira_filter import JIRAFilter
from athenian.api.models.web.jira_filter_with import JIRAFilterWith


class CalculatedJIRAMetricValues(Model, GranularityMixin):
    """Calculated JIRA metrics for a specific granularity."""

    granularity: str
    for_: Optional[JIRAFilter]
    jira_label: Optional[str]
    with_: Optional[JIRAFilterWith]
    values: List[CalculatedLinearMetricValues]

    attribute_map = {
        "with_": "with",
        "for_": "for",
    }
