from datetime import timedelta
from typing import List, Optional, Union

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.histogram_scale import HistogramScale
from athenian.api.models.web.interquartile import Interquartile
from athenian.api.models.web.jira_filter_with import JIRAFilterWith
from athenian.api.models.web.jira_metric_id import JIRAMetricID


class CalculatedJIRAHistogram(Model):
    """Calculated histogram over JIRA issue activities."""

    attribute_types = {
        "with_": Optional[JIRAFilterWith],
        "metric": str,
        "scale": str,
        "ticks": List[Union[float, timedelta]],
        "frequencies": List[int],
        "interquartile": Interquartile,
    }

    attribute_map = {
        "with_": "with",
        "metric": "metric",
        "scale": "scale",
        "ticks": "ticks",
        "frequencies": "frequencies",
        "interquartile": "interquartile",
    }

    def __init__(
        self,
        with_: Optional[JIRAFilterWith] = None,
        metric: Optional[str] = None,
        scale: Optional[str] = None,
        ticks: Optional[List[Union[float, timedelta]]] = None,
        frequencies: Optional[List[int]] = None,
        interquartile: Optional[Interquartile] = None,
    ):
        """CalculatedJIRAHistogram - a model defined in OpenAPI

        :param with_: The with_ of this CalculatedJIRAHistogram.
        :param metric: The metric of this CalculatedJIRAHistogram.
        :param scale: The scale of this CalculatedJIRAHistogram.
        :param ticks: The ticks of this CalculatedJIRAHistogram.
        :param frequencies: The frequencies of this CalculatedJIRAHistogram.
        :param interquartile: The interquartile of this CalculatedJIRAHistogram.
        """
        self._with_ = with_
        self._metric = metric
        self._scale = scale
        self._ticks = ticks
        self._frequencies = frequencies
        self._interquartile = interquartile

    @property
    def with_(self) -> Optional[JIRAFilterWith]:
        """Gets the with_ of this CalculatedJIRAHistogram.

        :return: The with_ of this CalculatedJIRAHistogram.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[JIRAFilterWith]):
        """Sets the with_ of this CalculatedJIRAHistogram.

        :param with_: The with_ of this CalculatedJIRAHistogram.
        """
        self._with_ = with_

    @property
    def metric(self) -> str:
        """Gets the metric of this CalculatedJIRAHistogram.

        :return: The metric of this CalculatedJIRAHistogram.
        """
        return self._metric

    @metric.setter
    def metric(self, metric):
        """Sets the metric of this CalculatedJIRAHistogram.

        :param metric: The metric of this CalculatedJIRAHistogram.
        :type metric: JIRAMetricID
        """
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")
        if metric not in JIRAMetricID:
            raise ValueError('"metrics" must be one of %s' % list(JIRAMetricID))

        self._metric = metric

    @property
    def scale(self) -> str:
        """Gets the scale of this CalculatedJIRAHistogram.

        :return: The scale of this CalculatedJIRAHistogram.
        """
        return self._scale

    @scale.setter
    def scale(self, scale: str):
        """Sets the scale of this CalculatedJIRAHistogram.

        :param scale: The scale of this CalculatedJIRAHistogram.
        """
        if scale is None:
            raise ValueError("Invalid value for `scale`, must not be `None`")
        if scale not in HistogramScale:
            raise ValueError('"scale" must be one of %s' % list(HistogramScale))

        self._scale = scale

    @property
    def ticks(self) -> List[float]:
        """Gets the ticks of this CalculatedJIRAHistogram.

        Series of horizontal bar borders aka X axis. Their count is `len(y) + 1` because there are
        `N` intervals between `(N + 1)` ticks.

        :return: The ticks of this CalculatedJIRAHistogram.
        """
        return self._ticks

    @ticks.setter
    def ticks(self, ticks: List[float]):
        """Sets the ticks of this CalculatedJIRAHistogram.

        Series of horizontal bar borders aka X axis. Their count is `len(y) + 1` because there are
        `N` intervals between `(N + 1)` ticks.

        :param ticks: The ticks of this CalculatedJIRAHistogram.
        """
        if ticks is None:
            raise ValueError("Invalid value for `ticks`, must not be `None`")

        self._ticks = ticks

    @property
    def frequencies(self) -> List[int]:
        """Gets the frequencies of this CalculatedJIRAHistogram.

        Series of histogram bar heights aka Y axis.

        :return: The frequencies of this CalculatedJIRAHistogram.
        """
        return self._frequencies

    @frequencies.setter
    def frequencies(self, frequencies: List[int]):
        """Sets the frequencies of this CalculatedJIRAHistogram.

        Series of histogram bar heights aka Y axis.

        :param frequencies: The frequencies of this CalculatedJIRAHistogram.
        """
        if frequencies is None:
            raise ValueError("Invalid value for `frequencies`, must not be `None`")

        self._frequencies = frequencies

    @property
    def interquartile(self) -> Interquartile:
        """Gets the interquartile of this CalculatedJIRAHistogram.

        :return: The interquartile of this CalculatedJIRAHistogram.
        """
        return self._interquartile

    @interquartile.setter
    def interquartile(self, interquartile: Interquartile):
        """Sets the interquartile of this CalculatedJIRAHistogram.

        :param interquartile: The interquartile of this CalculatedJIRAHistogram.
        """
        if interquartile is None:
            raise ValueError("Invalid value for `interquartile`, must not be `None`")

        self._interquartile = interquartile
