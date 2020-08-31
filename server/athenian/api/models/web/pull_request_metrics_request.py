from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.common_filter_properties import CommonFilterPropertiesMixin
from athenian.api.models.web.for_set import ForSet
from athenian.api.models.web.granularity import Granularity
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID
from athenian.api.models.web.quantiles import validate_quantiles


class PullRequestMetricsRequest(Model, CommonFilterPropertiesMixin):
    """TRequest for calculating metrics on top of pull requests data."""

    openapi_types = {
        "for_": List[ForSet],
        "metrics": List[PullRequestMetricID],
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "granularities": List[str],
        "quantiles": List[float],
        "account": int,
        "exclude_inactive": bool,
    }

    attribute_map = {
        "for_": "for",
        "metrics": "metrics",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "granularities": "granularities",
        "quantiles": "quantiles",
        "account": "account",
        "exclude_inactive": "exclude_inactive",
    }

    def __init__(
        self,
        for_: Optional[List[ForSet]] = None,
        metrics: Optional[List[PullRequestMetricID]] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        granularities: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        account: Optional[int] = None,
        exclude_inactive: Optional[bool] = None,
    ):
        """PullRequestMetricsRequest - a model defined in OpenAPI

        :param for_: The for_ of this PullRequestMetricsRequest.
        :param metrics: The metrics of this PullRequestMetricsRequest.
        :param date_from: The date_from of this PullRequestMetricsRequest.
        :param date_to: The date_to of this PullRequestMetricsRequest.
        :param timezone: The timezone of this PullRequestMetricsRequest.
        :param granularities: The granularities of this PullRequestMetricsRequest.
        :param quantiles: The quantiles of this PullRequestMetricsRequest.
        :param account: The account of this PullRequestMetricsRequest.
        :param exclude_inactive: The exclude_inactive of this PullRequestMetricsRequest.
        """
        self._for_ = for_
        self._metrics = metrics
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._granularities = granularities
        self._quantiles = quantiles
        self._account = account
        self._exclude_inactive = exclude_inactive

    @property
    def for_(self) -> List[ForSet]:
        """Gets the for_ of this PullRequestMetricsRequest.

        Sets of developers and repositories to calculate the metrics for.

        :return: The for_ of this PullRequestMetricsRequest.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: List[ForSet]):
        """Sets the for_ of this PullRequestMetricsRequest.

        Sets of developers and repositories to calculate the metrics for.

        :param for_: The for_ of this PullRequestMetricsRequest.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for`, must not be `None`")

        self._for_ = for_

    @property
    def metrics(self) -> List[PullRequestMetricID]:
        """Gets the metrics of this PullRequestMetricsRequest.

        Requested metric identifiers.

        :return: The metrics of this PullRequestMetricsRequest.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[PullRequestMetricID]):
        """Sets the metrics of this PullRequestMetricsRequest.

        Requested metric identifiers.

        :param metrics: The metrics of this PullRequestMetricsRequest.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")

        self._metrics = metrics

    @property
    def granularities(self) -> List[str]:
        """Gets the granularities of this PullRequestMetricsRequest.

        :return: The granularities of this PullRequestMetricsRequest.
        """
        return self._granularities

    @granularities.setter
    def granularities(self, granularities: List[str]):
        """Sets the granularities of this PullRequestMetricsRequest.

        :param granularities: The granularities of this PullRequestMetricsRequest.
        """
        if granularities is None:
            raise ValueError("Invalid value for `granularities`, must not be `None`")
        for i, g in enumerate(granularities):
            if not Granularity.format.match(g):
                raise ValueError(
                    'Invalid value for `granularity[%d]`: "%s"` does not match /%s/' %
                    (i, g, Granularity.format.pattern))

        self._granularities = granularities

    @property
    def quantiles(self) -> Optional[List[float]]:
        """Gets the quantiles of this PullRequestMetricsRequest.

        :return: The quantiles of this PullRequestMetricsRequest.
        """
        return self._quantiles

    @quantiles.setter
    def quantiles(self, quantiles: Optional[List[float]]):
        """Sets the quantiles of this PullRequestMetricsRequest.

        :param quantiles: The quantiles of this PullRequestMetricsRequest.
        """
        if quantiles is None:
            self._quantiles = None
            return
        validate_quantiles(quantiles)
        self._quantiles = quantiles

    @property
    def exclude_inactive(self) -> bool:
        """Gets the exclude_inactive of this PullRequestMetricsRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :return: The exclude_inactive of this PullRequestMetricsRequest.
        """
        return self._exclude_inactive

    @exclude_inactive.setter
    def exclude_inactive(self, exclude_inactive: bool):
        """Sets the exclude_inactive of this PullRequestMetricsRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :param exclude_inactive: The exclude_inactive of this PullRequestMetricsRequest.
        """
        self._exclude_inactive = exclude_inactive
