from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.developer_metric_id import DeveloperMetricID
from athenian.api.models.web.for_set import ForSet


class DeveloperMetricsRequest(Model):
    """Request for calculating metrics on top of developer activities."""

    def __init__(
        self,
        for_: Optional[List[ForSet]] = None,
        metrics: Optional[List[str]] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        account: Optional[int] = None,
    ):
        """DeveloperMetricsRequest - a model defined in OpenAPI

        :param for_: The for_ of this DeveloperMetricsRequest.
        :param metrics: The metrics of this DeveloperMetricsRequest.
        :param date_from: The date_from of this DeveloperMetricsRequest.
        :param date_to: The date_to of this DeveloperMetricsRequest.
        :param account: The account of this DeveloperMetricsRequest.
        """
        self.openapi_types = {
            "for_": List[ForSet],
            "metrics": List[str],
            "date_from": date,
            "date_to": date,
            "account": int,
        }

        self.attribute_map = {
            "for_": "for",
            "metrics": "metrics",
            "date_from": "date_from",
            "date_to": "date_to",
            "account": "account",
        }

        self._for_ = for_
        self._metrics = metrics
        self._date_from = date_from
        self._date_to = date_to
        self._account = account

    @property
    def for_(self) -> List[ForSet]:
        """Gets the for_ of this DeveloperMetricsRequest.

        Sets of developers and repositories to calculate the metrics for.

        :return: The for_ of this DeveloperMetricsRequest.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: List[ForSet]):
        """Sets the for_ of this DeveloperMetricsRequest.

        Sets of developers and repositories to calculate the metrics for.

        :param for_: The for_ of this DeveloperMetricsRequest.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for_`, must not be `None`")

        self._for_ = for_

    @property
    def metrics(self) -> List[str]:
        """Gets the metrics of this DeveloperMetricsRequest.

        Requested metric identifiers.

        :return: The metrics of this DeveloperMetricsRequest.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[str]):
        """Sets the metrics of this DeveloperMetricsRequest.

        Requested metric identifiers.

        :param metrics: The metrics of this DeveloperMetricsRequest.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        diff = set(metrics) - DeveloperMetricID.ALL
        if diff:
            raise ValueError("Unsupported values of `metrics`: %s" % diff)

        self._metrics = metrics

    @property
    def date_from(self) -> date:
        """Gets the date_from of this DeveloperMetricsRequest.

        Date from when to start measuring the metrics.

        :return: The date_from of this DeveloperMetricsRequest.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date):
        """Sets the date_from of this DeveloperMetricsRequest.

        Date from when to start measuring the metrics.

        :param date_from: The date_from of this DeveloperMetricsRequest.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this DeveloperMetricsRequest.

        Date up to which to measure the metrics.

        :return: The date_to of this DeveloperMetricsRequest.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date):
        """Sets the date_to of this DeveloperMetricsRequest.

        Date up to which to measure the metrics.

        :param date_to: The date_to of this DeveloperMetricsRequest.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def account(self) -> int:
        """Gets the account of this DeveloperMetricsRequest.

        Session account ID.

        :return: The account of this DeveloperMetricsRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this DeveloperMetricsRequest.

        Session account ID.

        :param account: The account of this DeveloperMetricsRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account
