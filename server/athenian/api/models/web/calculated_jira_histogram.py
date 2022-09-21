from datetime import timedelta
from typing import List, Optional, Union

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.histogram_scale import HistogramScale
from athenian.api.models.web.interquartile import Interquartile
from athenian.api.models.web.jira_filter_with import JIRAFilterWith
from athenian.api.models.web.jira_metric_id import JIRAMetricID


class CalculatedJIRAHistogram(Model):
    """Calculated histogram over JIRA issue activities."""

    with_: (Optional[JIRAFilterWith], "with")
    metric: str
    scale: str
    ticks: List[Union[float, timedelta]]
    frequencies: List[int]
    interquartile: Interquartile

    def validate_metric(self, metric: str) -> str:
        """Sets the metric of this CalculatedJIRAHistogram.

        :param metric: The metric of this CalculatedJIRAHistogram.
        """
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")
        if metric not in JIRAMetricID:
            raise ValueError('"metrics" must be one of %s' % list(JIRAMetricID))

        return metric

    def validate_scale(self, scale: str) -> str:
        """Sets the scale of this CalculatedJIRAHistogram.

        :param scale: The scale of this CalculatedJIRAHistogram.
        """
        if scale is None:
            raise ValueError("Invalid value for `scale`, must not be `None`")
        if scale not in HistogramScale:
            raise ValueError('"scale" must be one of %s' % list(HistogramScale))

        return scale
