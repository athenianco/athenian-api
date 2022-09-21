from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.code_check_metric_id import CodeCheckMetricID
from athenian.api.models.web.histogram_scale import HistogramScale


class CodeCheckHistogramDefinition(Model):
    """Histogram parameters: topic, bins."""

    metric: str
    scale: Optional[str]
    bins: Optional[int]
    ticks: Optional[List[float]]

    def validate_metric(self, metric: str) -> str:
        """Sets the metric of this CodeCheckHistogramDefinition.

        :param metric: The metric of this CodeCheckHistogramDefinition.
        """
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")
        if metric not in CodeCheckMetricID:
            raise ValueError("Invalid value for `metric`: %s" % metric)

        return metric

    def validate_scale(self, scale: Optional[str]) -> Optional[str]:
        """Sets the scale of this CodeCheckHistogramDefinition.

        :param scale: The scale of this CodeCheckHistogramDefinition.
        """
        if scale is not None and scale not in HistogramScale:
            raise ValueError("Invalid value of `scale`: %s" % scale)

        return scale

    def validate_bins(self, bins: Optional[int]) -> Optional[int]:
        """Sets the bins of this CodeCheckHistogramDefinition.

        Number of bars in the histogram. 0 or null means automatic.

        :param bins: The bins of this CodeCheckHistogramDefinition.
        """
        if bins is not None and bins < 0:
            raise ValueError(
                "Invalid value for `bins`, must be a value greater than or equal to `0`",
            )

        return bins
