from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.for_set import ForSet
from athenian.api.models.web.histogram_scale import HistogramScale
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID


class CalculatedPullRequestHistogram(Model):
    """Response from `/histograms/prs`."""

    openapi_types = {
        "for_": ForSet,
        "metric": str,
        "scale": str,
        "ticks": List[object],
        "frequencies": List[int],
    }

    attribute_map = {
        "for_": "for",
        "metric": "metric",
        "scale": "scale",
        "ticks": "ticks",
        "frequencies": "frequencies",
    }

    def __init__(
        self,
        for_: Optional[ForSet] = None,
        metric: Optional[str] = None,
        scale: Optional[str] = None,
        ticks: Optional[List[float]] = None,
        frequencies: Optional[List[int]] = None,
    ):
        """CalculatedPullRequestHistogram - a model defined in OpenAPI

        :param for_: The for_ of this CalculatedPullRequestHistogram.
        :param metric: The metric of this CalculatedPullRequestHistogram.
        :param scale: The scale of this CalculatedPullRequestHistogram.
        :param ticks: The ticks of this CalculatedPullRequestHistogram.
        :param frequencies: The frequencies of this CalculatedPullRequestHistogram.
        """
        self._for_ = for_
        self._metric = metric
        self._scale = scale
        self._ticks = ticks
        self._frequencies = frequencies

    @property
    def for_(self) -> ForSet:
        """Gets the for_ of this CalculatedPullRequestHistogram.

        :return: The for_ of this CalculatedPullRequestHistogram.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: ForSet):
        """Sets the for_ of this CalculatedPullRequestHistogram.

        :param for_: The for_ of this CalculatedPullRequestHistogram.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for_`, must not be `None`")

        self._for_ = for_

    @property
    def metric(self) -> str:
        """Gets the metric of this CalculatedPullRequestHistogram.

        :return: The metric of this CalculatedPullRequestHistogram.
        """
        return self._metric

    @metric.setter
    def metric(self, metric: str):
        """Sets the metric of this CalculatedPullRequestHistogram.

        :param metric: The metric of this CalculatedPullRequestHistogram.
        """
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")
        if metric not in PullRequestMetricID:
            raise ValueError('"metrics" must be one of %s' % list(PullRequestMetricID))

        self._metric = metric

    @property
    def scale(self) -> str:
        """Gets the scale of this CalculatedPullRequestHistogram.

        :return: The scale of this CalculatedPullRequestHistogram.
        """
        return self._scale

    @scale.setter
    def scale(self, scale: str):
        """Sets the scale of this CalculatedPullRequestHistogram.

        :param scale: The scale of this CalculatedPullRequestHistogram.
        """
        if scale is None:
            raise ValueError("Invalid value for `scale`, must not be `None`")
        if scale not in HistogramScale:
            raise ValueError('"scale" must be one of %s' % list(HistogramScale))

        self._scale = scale

    @property
    def ticks(self) -> List[object]:
        """Gets the ticks of this CalculatedPullRequestHistogram.

        Series of horizontal bar borders aka X axis. Their count is `len(y) + 1` because there are
        `N` intervals between `(N + 1)` ticks.

        :return: The ticks of this CalculatedPullRequestHistogram.
        """
        return self._ticks

    @ticks.setter
    def ticks(self, ticks: List[object]):
        """Sets the ticks of this CalculatedPullRequestHistogram.

        Series of horizontal bar borders aka X axis. Their count is `len(y) + 1` because there are
        `N` intervals between `(N + 1)` ticks.

        :param ticks: The ticks of this CalculatedPullRequestHistogram.
        """
        if ticks is None:
            raise ValueError("Invalid value for `ticks`, must not be `None`")

        self._ticks = ticks

    @property
    def frequencies(self) -> List[int]:
        """Gets the frequencies of this CalculatedPullRequestHistogram.

        Series of histogram bar heights aka Y axis.

        :return: The frequencies of this CalculatedPullRequestHistogram.
        """
        return self._frequencies

    @frequencies.setter
    def frequencies(self, frequencies: List[int]):
        """Sets the frequencies of this CalculatedPullRequestHistogram.

        Series of histogram bar heights aka Y axis.

        :param frequencies: The frequencies of this CalculatedPullRequestHistogram.
        """
        if frequencies is None:
            raise ValueError("Invalid value for `frequencies`, must not be `None`")

        self._frequencies = frequencies
