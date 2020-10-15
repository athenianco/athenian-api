from typing import Any, List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.histogram_scale import HistogramScale
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID
from athenian.api.serialization import deserialize_timedelta


class HistogramDefinition(Model):
    """Histogram parameters for each wanted topic."""

    openapi_types = {
        "metric": PullRequestMetricID,
        "scale": Optional[HistogramScale],
        "bins": Optional[int],
        "ticks": Optional[List[object]],
    }

    attribute_map = {
        "metric": "metric",
        "scale": "scale",
        "bins": "bins",
        "ticks": "ticks",
    }

    def __init__(
        self,
        metric: Optional[PullRequestMetricID] = None,
        scale: Optional[HistogramScale] = None,
        bins: Optional[int] = None,
        ticks: Optional[List[object]] = None,
    ):
        """
        HistogramDefinition - a model defined in OpenAPI

        :param metric: The metric of this HistogramDefinition.
        :param scale: The scale of this HistogramDefinition.
        :param bins: The bins of this HistogramDefinition.
        :param ticks: The ticks of this HistogramDefinition.
        """
        self._metric = metric
        self._scale = scale
        self._bins = bins
        self._ticks = ticks

    @property
    def metric(self) -> PullRequestMetricID:
        """Gets the metric of this HistogramDefinition.

        :return: The metric of this HistogramDefinition.
        """
        return self._metric

    @metric.setter
    def metric(self, metric: PullRequestMetricID):
        """Sets the metric of this HistogramDefinition.

        :param metric: The metric of this HistogramDefinition.
        """
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")

        self._metric = metric

    @property
    def scale(self) -> Optional[str]:
        """Gets the scale of this HistogramDefinition.

        Histogram's X axis scale.

        :return: The scale of this HistogramDefinition.
        """
        return self._scale

    @scale.setter
    def scale(self, scale: Optional[str]):
        """Sets the scale of this HistogramDefinition.

        Histogram's X axis scale.

        :param scale: The scale of this HistogramDefinition.
        """
        if scale is not None and scale not in HistogramScale:
            raise ValueError('"scale" must be one of %s' % list(HistogramScale))

        self._scale = scale

    @property
    def bins(self) -> Optional[int]:
        """Gets the bins of this HistogramDefinition.

        Number of bars in the histogram. 0 or null means automatic.

        :return: The bins of this HistogramDefinition.
        """
        return self._bins

    @bins.setter
    def bins(self, bins: Optional[int]):
        """Sets the bins of this HistogramDefinition.

        Number of bars in the histogram. 0 or null means automatic.

        :param bins: The bins of this HistogramDefinition.
        """
        if bins is not None and bins < 0:
            raise ValueError(
                "Invalid value for `bins`, must be a value greater than or equal to `0`")

        self._bins = bins

    @property
    def ticks(self) -> Optional[List[Any]]:
        """Gets the ticks of this HistogramDefinition.

        Alternatively to `bins` and `scale`, set the X axis bar borders manually.
        Only one of two may be specified. The ticks are automatically prepended
        the distribution minimum and appended the distribution maximum.

        :return: The ticks of this HistogramDefinition.
        """
        return self._ticks

    @ticks.setter
    def ticks(self, ticks: Optional[List[Any]]):
        """Sets the ticks of this HistogramDefinition.

        Alternatively to `bins` and `scale`, set the X axis bar borders manually.
        Only one of two may be specified. The ticks are automatically prepended
        the distribution minimum and appended the distribution maximum.

        :param ticks: The ticks of this HistogramDefinition.
        """
        if ticks is not None and len(ticks) == 0:
            raise ValueError("`ticks` must contain at least one element")

        if ticks and isinstance(ticks[0], str):
            ticks = [deserialize_timedelta(s) for s in ticks]

        self._ticks = ticks
