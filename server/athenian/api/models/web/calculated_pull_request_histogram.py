from datetime import timedelta
from typing import List, Union

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.for_set_pull_requests import ForSetPullRequests
from athenian.api.models.web.histogram_scale import HistogramScale
from athenian.api.models.web.interquartile import Interquartile
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID


class CalculatedPullRequestHistogram(Model):
    """Response from `/histograms/pull_requests`."""

    for_: (ForSetPullRequests, "for")
    metric: str
    scale: str
    ticks: List[Union[float, timedelta]]
    frequencies: List[int]
    interquartile: Interquartile

    def validate_metric(self, metric: str) -> str:
        """Sets the metric of this CalculatedPullRequestHistogram.

        :param metric: The metric of this CalculatedPullRequestHistogram.
        """
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")
        if metric not in PullRequestMetricID:
            raise ValueError('"metrics" must be one of %s' % list(PullRequestMetricID))

        return metric

    def validate_scale(self, scale: str) -> str:
        """Sets the scale of this CalculatedPullRequestHistogram.

        :param scale: The scale of this CalculatedPullRequestHistogram.
        """
        if scale is None:
            raise ValueError("Invalid value for `scale`, must not be `None`")
        if scale not in HistogramScale:
            raise ValueError('"scale" must be one of %s' % list(HistogramScale))

        return scale
