from typing import List

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.code_check_metric_id import CodeCheckMetricID
from athenian.api.models.web.for_set_code_checks import _CalculatedCodeCheckCommon
from athenian.api.models.web.histogram_scale import HistogramScale
from athenian.api.models.web.interquartile import Interquartile


class _CalculatedCodeCheckHistogram(Model, sealed=False):
    """Calculated histogram over code checks (CI)."""

    metric: str
    scale: str
    ticks: List[float]
    frequencies: List[int]
    interquartile: Interquartile

    def validate_metric(self, metric: str) -> str:
        """Sets the metric of this CalculatedCodeCheckHistogram.

        :param metric: The metric of this CalculatedCodeCheckHistogram.
        """
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")
        if metric not in CodeCheckMetricID:
            raise ValueError("Invalid value of the code check metric: %s" % metric)

        return metric

    def validate_scale(self, scale: str):
        """Sets the scale of this CalculatedCodeCheckHistogram.

        :param scale: The scale of this CalculatedCodeCheckHistogram.
        """
        if scale is None:
            raise ValueError("Invalid value for `scale`, must not be `None`")
        if scale not in HistogramScale:
            raise ValueError("Invalid histogram scale value: %s" % scale)

        return scale


CalculatedCodeCheckHistogram = AllOf(
    _CalculatedCodeCheckHistogram,
    _CalculatedCodeCheckCommon,
    name="CalculatedCodeCheckHistogram",
    module=__name__,
)
