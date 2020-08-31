from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.common_filter_properties import CommonFilterPropertiesMixin
from athenian.api.models.web.for_set import ForSet
from athenian.api.models.web.histogram_scale import HistogramScale
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID
from athenian.api.models.web.quantiles import validate_quantiles


class PullRequestHistogramsRequest(Model, CommonFilterPropertiesMixin):
    """Request of `/histograms/prs`."""

    openapi_types = {
        "for_": List[ForSet],
        "metrics": List[str],
        "scale": str,
        "bins": int,
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "exclude_inactive": bool,
        "quantiles": List[float],
        "account": int,
    }

    attribute_map = {
        "for_": "for",
        "metrics": "metrics",
        "scale": "scale",
        "bins": "bins",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "exclude_inactive": "exclude_inactive",
        "quantiles": "quantiles",
        "account": "account",
    }

    def __init__(
        self,
        for_: Optional[List[ForSet]] = None,
        metrics: Optional[List[str]] = None,
        scale: Optional[str] = None,
        bins: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        exclude_inactive: Optional[bool] = None,
        quantiles: Optional[List[float]] = None,
        account: Optional[int] = None,
    ):
        """PullRequestHistogramsRequest - a model defined in OpenAPI

        :param for_: The for_ of this PullRequestHistogramsRequest.
        :param metrics: The metrics of this PullRequestHistogramsRequest.
        :param scale: The scale of this PullRequestHistogramsRequest.
        :param bins: The bins of this PullRequestHistogramsRequest.
        :param date_from: The date_from of this PullRequestHistogramsRequest.
        :param date_to: The date_to of this PullRequestHistogramsRequest.
        :param timezone: The timezone of this PullRequestHistogramsRequest.
        :param exclude_inactive: The exclude_inactive of this PullRequestHistogramsRequest.
        :param quantiles: The quantiles of this PullRequestHistogramsRequest.
        :param account: The account of this PullRequestHistogramsRequest.
        """
        self._for_ = for_
        self._metrics = metrics
        self._scale = scale
        self._bins = bins
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._exclude_inactive = exclude_inactive
        self._quantiles = quantiles
        self._account = account

    @property
    def for_(self) -> List[ForSet]:
        """Gets the for_ of this PullRequestHistogramsRequest.

        Sets of developers and repositories for which to calculate the histograms.
        The aggregation is `AND` between repositories and developers. The aggregation is `OR`
        inside both repositories and developers.

        :return: The for_ of this PullRequestHistogramsRequest.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: List[ForSet]):
        """Sets the for_ of this PullRequestHistogramsRequest.

        Sets of developers and repositories for which to calculate the histograms.
        The aggregation is `AND` between repositories and developers. The aggregation is `OR`
        inside both repositories and developers.

        :param for_: The for_ of this PullRequestHistogramsRequest.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for_`, must not be `None`")

        self._for_ = for_

    @property
    def metrics(self) -> List[str]:
        """Gets the metrics of this PullRequestHistogramsRequest.

        List of desired histogram topics.

        :return: The metrics of this PullRequestHistogramsRequest.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[str]):
        """Sets the metrics of this PullRequestHistogramsRequest.

        List of desired histogram topics.

        :param metrics: The metrics of this PullRequestHistogramsRequest.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for m in metrics:
            if m not in PullRequestMetricID:
                raise ValueError('"%s" is not one of %s' % (m, list(PullRequestMetricID)))

        self._metrics = metrics

    @property
    def scale(self) -> str:
        """Gets the scale of this PullRequestHistogramsRequest.

        :return: The scale of this PullRequestHistogramsRequest.
        """
        return self._scale

    @scale.setter
    def scale(self, scale: str):
        """Sets the scale of this PullRequestHistogramsRequest.

        :param scale: The scale of this PullRequestHistogramsRequest.
        """
        if scale is None:
            raise ValueError("Invalid value for `scale`, must not be `None`")
        if scale not in HistogramScale:
            raise ValueError('"scale" must be one of %s' % list(HistogramScale))

        self._scale = scale

    @property
    def bins(self) -> int:
        """Gets the bins of this PullRequestHistogramsRequest.

        Number of bars in the histogram. 0 means automatic.

        :return: The bins of this PullRequestHistogramsRequest.
        """
        return self._bins

    @bins.setter
    def bins(self, bins: int):
        """Sets the bins of this PullRequestHistogramsRequest.

        Number of bars in the histogram. 0 means automatic.

        :param bins: The bins of this PullRequestHistogramsRequest.
        """
        if bins is not None and bins < 0:
            raise ValueError(
                "Invalid value for `bins`, must be a value greater than or equal to `0`")

        self._bins = bins

    @property
    def exclude_inactive(self) -> bool:
        """Gets the exclude_inactive of this PullRequestHistogramsRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :return: The exclude_inactive of this PullRequestHistogramsRequest.
        """
        return self._exclude_inactive

    @exclude_inactive.setter
    def exclude_inactive(self, exclude_inactive: bool):
        """Sets the exclude_inactive of this PullRequestHistogramsRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :param exclude_inactive: The exclude_inactive of this PullRequestHistogramsRequest.
        """
        if exclude_inactive is None:
            raise ValueError("Invalid value for `exclude_inactive`, must not be `None`")

        self._exclude_inactive = exclude_inactive

    @property
    def quantiles(self) -> Optional[List[float]]:
        """Gets the quantiles of this PullRequestHistogramsRequest.

        :return: The quantiles of this PullRequestHistogramsRequest.
        """
        return self._quantiles

    @quantiles.setter
    def quantiles(self, quantiles: Optional[List[float]]):
        """Sets the quantiles of this PullRequestHistogramsRequest.

        :param quantiles: The quantiles of this PullRequestHistogramsRequest.
        """
        if quantiles is None:
            self._quantiles = None
            return
        validate_quantiles(quantiles)
        self._quantiles = quantiles
