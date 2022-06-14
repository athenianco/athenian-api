from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import CommonMetricsProperties
from athenian.api.models.web.for_set_pull_requests import ForSetPullRequests
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID


class _PullRequestMetricsRequest(Model, sealed=False):
    """Request for calculating metrics on top of pull requests data."""

    attribute_types = {
        "for_": List[ForSetPullRequests],
        "metrics": List[PullRequestMetricID],
        "exclude_inactive": bool,
        "fresh": bool,
    }

    attribute_map = {
        "for_": "for",
        "metrics": "metrics",
        "exclude_inactive": "exclude_inactive",
        "fresh": "fresh",
    }

    def __init__(
        self,
        for_: Optional[List[ForSetPullRequests]] = None,
        metrics: Optional[List[PullRequestMetricID]] = None,
        exclude_inactive: Optional[bool] = None,
        fresh: Optional[bool] = None,
    ):
        """PullRequestMetricsRequest - a model defined in OpenAPI

        :param for_: The for_ of this PullRequestMetricsRequest.
        :param metrics: The metrics of this PullRequestMetricsRequest.
        :param exclude_inactive: The exclude_inactive of this PullRequestMetricsRequest.
        :param fresh: The fresh of this PullRequestMetricsRequest.
        """
        self._for_ = for_
        self._metrics = metrics
        self._exclude_inactive = exclude_inactive
        self._fresh = fresh

    @property
    def for_(self) -> List[ForSetPullRequests]:
        """Gets the for_ of this PullRequestMetricsRequest.

        Sets of developers and repositories to calculate the metrics for.

        :return: The for_ of this PullRequestMetricsRequest.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: List[ForSetPullRequests]):
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

    @property
    def fresh(self) -> bool:
        """Gets the fresh of this PullRequestMetricsRequest.

        Force metrics calculation on the most up to date data.

        :return: The fresh of this PullRequestMetricsRequest.
        """
        return self._fresh

    @fresh.setter
    def fresh(self, fresh: bool):
        """Sets the fresh of this PullRequestMetricsRequest.

        Force metrics calculation on the most up to date data.

        :param fresh: The fresh of this PullRequestMetricsRequest.
        """
        self._fresh = fresh


PullRequestMetricsRequest = AllOf(
    _PullRequestMetricsRequest,
    CommonFilterProperties,
    CommonMetricsProperties,
    name="PullRequestMetricsRequest",
    module=__name__,
)
