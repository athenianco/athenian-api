from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.code_check_metric_id import CodeCheckMetricID
from athenian.api.models.web.for_set_code_checks import _CalculatedCodeCheckCommon
from athenian.api.models.web.histogram_scale import HistogramScale
from athenian.api.models.web.interquartile import Interquartile


class _CalculatedCodeCheckHistogram(Model, sealed=False):
    """Calculated histogram over code checks (CI)."""

    attribute_types = {
        "metric": str,
        "scale": str,
        "ticks": List[float],
        "frequencies": List[int],
        "interquartile": Interquartile,
    }

    attribute_map = {
        "metric": "metric",
        "scale": "scale",
        "ticks": "ticks",
        "frequencies": "frequencies",
        "interquartile": "interquartile",
    }

    def __init__(
        self,
        metric: Optional[str] = None,
        scale: Optional[str] = None,
        ticks: Optional[List[float]] = None,
        frequencies: Optional[List[int]] = None,
        interquartile: Optional[Interquartile] = None,
    ):
        """CalculatedCodeCheckHistogram - a model defined in OpenAPI

        :param metric: The metric of this CalculatedCodeCheckHistogram.
        :param scale: The scale of this CalculatedCodeCheckHistogram.
        :param ticks: The ticks of this CalculatedCodeCheckHistogram.
        :param frequencies: The frequencies of this CalculatedCodeCheckHistogram.
        :param interquartile: The interquartile of this CalculatedCodeCheckHistogram.
        """
        self._metric = metric
        self._scale = scale
        self._ticks = ticks
        self._frequencies = frequencies
        self._interquartile = interquartile

    @property
    def metric(self) -> str:
        """Gets the metric of this CalculatedCodeCheckHistogram.

        :return: The metric of this CalculatedCodeCheckHistogram.
        """
        return self._metric

    @metric.setter
    def metric(self, metric: str):
        """Sets the metric of this CalculatedCodeCheckHistogram.

        :param metric: The metric of this CalculatedCodeCheckHistogram.
        """
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")
        if metric not in CodeCheckMetricID:
            raise ValueError("Invalid value of the code check metric: %s" % metric)

        self._metric = metric

    @property
    def scale(self) -> str:
        """Gets the scale of this CalculatedCodeCheckHistogram.

        :return: The scale of this CalculatedCodeCheckHistogram.
        """
        return self._scale

    @scale.setter
    def scale(self, scale: str):
        """Sets the scale of this CalculatedCodeCheckHistogram.

        :param scale: The scale of this CalculatedCodeCheckHistogram.
        """
        if scale is None:
            raise ValueError("Invalid value for `scale`, must not be `None`")
        if scale not in HistogramScale:
            raise ValueError("Invalid histogram scale value: %s" % scale)

        self._scale = scale

    @property
    def ticks(self) -> List[float]:
        """Gets the ticks of this CalculatedCodeCheckHistogram.

        Series of horizontal bar borders aka X axis. Their count is `len(y) + 1` because there
        are `N` intervals between `(N + 1)` ticks.

        :return: The ticks of this CalculatedCodeCheckHistogram.
        """
        return self._ticks

    @ticks.setter
    def ticks(self, ticks: List[float]):
        """Sets the ticks of this CalculatedCodeCheckHistogram.

        Series of horizontal bar borders aka X axis. Their count is `len(y) + 1` because there
        are `N` intervals between `(N + 1)` ticks.

        :param ticks: The ticks of this CalculatedCodeCheckHistogram.
        """
        if ticks is None:
            raise ValueError("Invalid value for `ticks`, must not be `None`")

        self._ticks = ticks

    @property
    def frequencies(self) -> List[int]:
        """Gets the frequencies of this CalculatedCodeCheckHistogram.

        Series of histogram bar heights aka Y axis.

        :return: The frequencies of this CalculatedCodeCheckHistogram.
        """
        return self._frequencies

    @frequencies.setter
    def frequencies(self, frequencies: List[int]):
        """Sets the frequencies of this CalculatedCodeCheckHistogram.

        Series of histogram bar heights aka Y axis.

        :param frequencies: The frequencies of this CalculatedCodeCheckHistogram.
        """
        if frequencies is None:
            raise ValueError("Invalid value for `frequencies`, must not be `None`")

        self._frequencies = frequencies

    @property
    def interquartile(self) -> Interquartile:
        """Gets the interquartile of this CalculatedCodeCheckHistogram.

        :return: The interquartile of this CalculatedCodeCheckHistogram.
        """
        return self._interquartile

    @interquartile.setter
    def interquartile(self, interquartile: Interquartile):
        """Sets the interquartile of this CalculatedCodeCheckHistogram.

        :param interquartile: The interquartile of this CalculatedCodeCheckHistogram.
        """
        if interquartile is None:
            raise ValueError("Invalid value for `interquartile`, must not be `None`")

        self._interquartile = interquartile


CalculatedCodeCheckHistogram = AllOf(
    _CalculatedCodeCheckHistogram,
    _CalculatedCodeCheckCommon,
    name="CalculatedCodeCheckHistogram",
    module=__name__,
)
