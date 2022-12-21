from typing import Dict, List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import CalculatedLinearMetricValues
from athenian.api.models.web.granularity import GranularityMixin
from athenian.api.models.web.release_metric_id import ReleaseMetricID
from athenian.api.models.web.release_with import ReleaseWith


class CalculatedReleaseMetric(Model, GranularityMixin):
    """Response from `/metrics/releases`."""

    for_: (Optional[List[str]], "for")
    with_: (Optional[ReleaseWith], "with")
    matches: Dict[str, str]
    metrics: List[str]
    granularity: str
    values: List[CalculatedLinearMetricValues]

    def validate_metrics(self, metrics: list[str]) -> list[str]:
        """Sets the metrics of this CalculatedReleaseMetric.

        :param metrics: The metrics of this CalculatedReleaseMetric.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for i, metric in enumerate(metrics):
            if metric not in ReleaseMetricID:
                raise ValueError(
                    "`metrics[%d]` = '%s' must be one of %s" % (i, metric, list(ReleaseMetricID)),
                )

        return metrics
