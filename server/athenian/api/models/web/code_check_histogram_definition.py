from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.code_check_metric_id import CodeCheckMetricID
from athenian.api.models.web.histogram_scale import HistogramScale


class CodeCheckHistogramDefinition(Model):
    """Histogram parameters: topic, bins."""

    attribute_types = {
        "metric": str,
        "scale": Optional[str],
        "bins": Optional[int],
        "ticks": Optional[List[float]],
    }

    attribute_map = {
        "metric": "metric",
        "scale": "scale",
        "bins": "bins",
        "ticks": "ticks",
    }

    def __init__(
        self,
        metric: Optional[str] = None,
        scale: Optional[str] = None,
        bins: Optional[int] = None,
        ticks: Optional[List[float]] = None,
    ):
        """CodeCheckHistogramDefinition - a model defined in OpenAPI

        :param metric: The metric of this CodeCheckHistogramDefinition.
        :param scale: The scale of this CodeCheckHistogramDefinition.
        :param bins: The bins of this CodeCheckHistogramDefinition.
        :param ticks: The ticks of this CodeCheckHistogramDefinition.
        """
        self._metric = metric
        self._scale = scale
        self._bins = bins
        self._ticks = ticks

    @property
    def metric(self) -> str:
        """Gets the metric of this CodeCheckHistogramDefinition.

        :return: The metric of this CodeCheckHistogramDefinition.
        """
        return self._metric

    @metric.setter
    def metric(self, metric: str):
        """Sets the metric of this CodeCheckHistogramDefinition.

        :param metric: The metric of this CodeCheckHistogramDefinition.
        """
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")
        if metric not in CodeCheckMetricID:
            raise ValueError("Invalid value for `metric`: %s" % metric)

        self._metric = metric

    @property
    def scale(self) -> Optional[str]:
        """Gets the scale of this CodeCheckHistogramDefinition.

        :return: The scale of this CodeCheckHistogramDefinition.
        """
        return self._scale

    @scale.setter
    def scale(self, scale: Optional[str]):
        """Sets the scale of this CodeCheckHistogramDefinition.

        :param scale: The scale of this CodeCheckHistogramDefinition.
        """
        if scale is not None and scale not in HistogramScale:
            raise ValueError("Invalid value of `scale`: %s" % scale)

        self._scale = scale

    @property
    def bins(self) -> Optional[int]:
        """Gets the bins of this CodeCheckHistogramDefinition.

        Number of bars in the histogram. 0 or null means automatic.

        :return: The bins of this CodeCheckHistogramDefinition.
        """
        return self._bins

    @bins.setter
    def bins(self, bins: Optional[int]):
        """Sets the bins of this CodeCheckHistogramDefinition.

        Number of bars in the histogram. 0 or null means automatic.

        :param bins: The bins of this CodeCheckHistogramDefinition.
        """
        if bins is not None and bins < 0:
            raise ValueError(
                "Invalid value for `bins`, must be a value greater than or equal to `0`")

        self._bins = bins

    @property
    def ticks(self) -> Optional[List[float]]:
        """Gets the ticks of this CodeCheckHistogramDefinition.

        Alternatively to `bins` and `scale`, set the X axis bar borders manually. Only one of two
        may be specified. The ticks are automatically prepended the distribution minimum and
        appended the distribution maximum.

        :return: The ticks of this CodeCheckHistogramDefinition.
        """
        return self._ticks

    @ticks.setter
    def ticks(self, ticks: Optional[List[float]]):
        """Sets the ticks of this CodeCheckHistogramDefinition.

        Alternatively to `bins` and `scale`, set the X axis bar borders manually. Only one of two
        may be specified. The ticks are automatically prepended the distribution minimum and
        appended the distribution maximum.

        :param ticks: The ticks of this CodeCheckHistogramDefinition.
        """
        self._ticks = ticks
