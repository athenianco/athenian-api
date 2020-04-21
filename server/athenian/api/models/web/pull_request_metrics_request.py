from datetime import date, datetime
from typing import List, Optional

from athenian.api.models.web import Granularity
from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.for_set import ForSet
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID


class PullRequestMetricsRequest(Model):
    """TRequest for calculating metrics on top of pull requests data."""

    def __init__(
        self,
        for_: Optional[List[ForSet]] = None,
        metrics: Optional[List[PullRequestMetricID]] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        granularities: Optional[List[str]] = None,
        account: Optional[int] = None,
    ):
        """PullRequestMetricsRequest - a model defined in OpenAPI

        :param for_: The for_ of this PullRequestMetricsRequest.
        :param metrics: The metrics of this PullRequestMetricsRequest.
        :param date_from: The date_from of this PullRequestMetricsRequest.
        :param date_to: The date_to of this PullRequestMetricsRequest.
        :param granularities: The granularities of this PullRequestMetricsRequest.
        :param account: The account of this PullRequestMetricsRequest.
        """
        self.openapi_types = {
            "for_": List[ForSet],
            "metrics": List[PullRequestMetricID],
            "date_from": date,
            "date_to": date,
            "granularities": List[str],
            "account": int,
        }

        self.attribute_map = {
            "for_": "for",
            "metrics": "metrics",
            "date_from": "date_from",
            "date_to": "date_to",
            "granularities": "granularities",
            "account": "account",
        }

        self._for_ = for_
        self._metrics = metrics
        self._date_from = date_from
        self._date_to = date_to
        self._granularities = granularities
        self._account = account

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
    def date_from(self) -> date:
        """Gets the date_from of this PullRequestMetricsRequest.

        The date from when to start measuring the metrics.

        :return: The date_from of this PullRequestMetricsRequest.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date):
        """Sets the date_from of this PullRequestMetricsRequest.

        The date from when to start measuring the metrics.

        :param date_from: The date_from of this PullRequestMetricsRequest.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        if isinstance(date_from, datetime):
            date_from = date_from.date()
        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this PullRequestMetricsRequest.

        The date up to which to measure the metrics.

        :return: The date_to of this PullRequestMetricsRequest.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date):
        """Sets the date_to of this PullRequestMetricsRequest.

        The date up to which to measure the metrics.

        :param date_to: The date_to of this PullRequestMetricsRequest.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        if isinstance(date_to, datetime):
            date_to = date_to.date()
        self._date_to = date_to

    @property
    def granularities(self) -> List[str]:
        """Gets the granularities of this PullRequestMetricsRequest.

        :return: The granularities of this PullRequestMetricsRequest.
        """
        return self._granularities

    @granularities.setter
    def granularities(self, granularities: List[str]):
        """Sets the granularity of this PullRequestMetricsRequest.

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
    def account(self) -> int:
        """Gets the account of this PullRequestMetricsRequest.

        :return: The account of this PullRequestMetricsRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this PullRequestMetricsRequest.

        :param account: The account of this PullRequestMetricsRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account
