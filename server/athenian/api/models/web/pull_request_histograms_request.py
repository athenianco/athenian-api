from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import QuantilesMixin
from athenian.api.models.web.for_set_pull_requests import ForSetPullRequests
from athenian.api.models.web.pull_request_histogram_definition import \
    PullRequestHistogramDefinition


class _PullRequestHistogramsRequest(Model, QuantilesMixin, sealed=False):
    """Request of `/histograms/pull_requests`."""

    attribute_types = {
        "for_": List[ForSetPullRequests],
        "histograms": List[PullRequestHistogramDefinition],
        "exclude_inactive": bool,
        "quantiles": List[float],
        "fresh": bool,
    }

    attribute_map = {
        "for_": "for",
        "histograms": "histograms",
        "exclude_inactive": "exclude_inactive",
        "quantiles": "quantiles",
        "fresh": "fresh",
    }

    def __init__(
        self,
        for_: Optional[List[ForSetPullRequests]] = None,
        histograms: Optional[List[PullRequestHistogramDefinition]] = None,
        exclude_inactive: Optional[bool] = None,
        quantiles: Optional[List[float]] = None,
        fresh: Optional[bool] = None,
    ):
        """PullRequestHistogramsRequest - a model defined in OpenAPI

        :param for_: The for_ of this PullRequestHistogramsRequest.
        :param histograms: The histograms of this PullRequestHistogramsRequest.
        :param exclude_inactive: The exclude_inactive of this PullRequestHistogramsRequest.
        :param quantiles: The quantiles of this PullRequestHistogramsRequest.
        :param fresh: The fresh of this PullRequestHistogramsRequest.
        """
        self._for_ = for_
        self._histograms = histograms
        self._exclude_inactive = exclude_inactive
        self._quantiles = quantiles
        self._fresh = fresh

    @property
    def for_(self) -> List[ForSetPullRequests]:
        """Gets the for_ of this PullRequestHistogramsRequest.

        Sets of developers and repositories for which to calculate the histograms.
        The aggregation is `AND` between repositories and developers. The aggregation is `OR`
        inside both repositories and developers.

        :return: The for_ of this PullRequestHistogramsRequest.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: List[ForSetPullRequests]):
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
    def histograms(self) -> List[PullRequestHistogramDefinition]:
        """Gets the histograms of this PullRequestHistogramsRequest.

        Histogram parameters for each wanted topic.

        :return: The histograms of this PullRequestHistogramsRequest.
        """
        return self._histograms

    @histograms.setter
    def histograms(self, histograms: List[PullRequestHistogramDefinition]):
        """Sets the histograms of this PullRequestHistogramsRequest.

        Histogram parameters for each wanted topic.

        :param histograms: The histograms of this PullRequestHistogramsRequest.
        """
        self._histograms = histograms

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
    def fresh(self) -> bool:
        """Gets the fresh of this PullRequestHistogramsRequest.

        Force histograms calculation on the most up to date data.

        :return: The fresh of this PullRequestHistogramsRequest.
        """
        return self._fresh

    @fresh.setter
    def fresh(self, fresh: bool):
        """Sets the fresh of this PullRequestHistogramsRequest.

        Force histograms calculation on the most up to date data.

        :param fresh: The fresh of this PullRequestHistogramsRequest.
        """
        self._fresh = fresh


PullRequestHistogramsRequest = AllOf(_PullRequestHistogramsRequest, CommonFilterProperties,
                                     name="PullRequestHistogramsRequest", module=__name__)
