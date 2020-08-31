from datetime import date
from typing import List, Optional

from athenian.api.models.web import Granularity, validate_quantiles
from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.release_metric_id import ReleaseMetricID


class ReleaseMetricsRequest(Model):
    """Request of `/metrics/releases` to calculate metrics on releases."""

    openapi_types = {
        "for_": List[List[str]],
        "metrics": List[str],
        "date_from": date,
        "date_to": date,
        "granularities": List[str],
        "quantiles": List[float],
        "timezone": int,
        "account": int,
    }

    attribute_map = {
        "for_": "for",
        "metrics": "metrics",
        "date_from": "date_from",
        "date_to": "date_to",
        "granularities": "granularities",
        "quantiles": "quantiles",
        "timezone": "timezone",
        "account": "account",
    }

    def __init__(
        self,
        for_: Optional[List[List[str]]] = None,
        metrics: Optional[List[str]] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        granularities: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        timezone: Optional[int] = None,
        account: Optional[int] = None,
    ):
        """ReleaseMetricsRequest - a model defined in OpenAPI

        :param for_: The for of this ReleaseMetricsRequest.
        :param metrics: The metrics of this ReleaseMetricsRequest.
        :param date_from: The date_from of this ReleaseMetricsRequest.
        :param date_to: The date_to of this ReleaseMetricsRequest.
        :param granularities: The granularities of this ReleaseMetricsRequest.
        :param timezone: The timezone of this ReleaseMetricsRequest.
        :param account: The account of this ReleaseMetricsRequest.
        """
        self._for_ = for_
        self._metrics = metrics
        self._date_from = date_from
        self._date_to = date_to
        self._granularities = granularities
        self._quantiles = quantiles
        self._timezone = timezone
        self._account = account

    @property
    def for_(self) -> List[List]:
        """Gets the for_ of this ReleaseMetricsRequest.

        List of repository groups for which to calculate the metrics.

        :return: The for_ of this ReleaseMetricsRequest.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: List[List]):
        """Sets the for_ of this ReleaseMetricsRequest.

        List of repository groups for which to calculate the metrics.

        :param for_: The for_ of this ReleaseMetricsRequest.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for_`, must not be `None`")

        self._for_ = for_

    @property
    def metrics(self) -> List[str]:
        """Gets the metrics of this ReleaseMetricsRequest.

        List of desired release metrics.

        :return: The metrics of this ReleaseMetricsRequest.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[str]):
        """Sets the metrics of this ReleaseMetricsRequest.

        List of desired release metrics.

        :param metrics: The metrics of this ReleaseMetricsRequest.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for i, metric in enumerate(metrics):
            if metric not in ReleaseMetricID:
                raise ValueError("`metrics[%d]` = '%s' must be one of %s" % (
                    i, metric, list(ReleaseMetricID)))

        self._metrics = metrics

    @property
    def date_from(self) -> date:
        """Gets the date_from of this ReleaseMetricsRequest.

        Date from when to start measuring the metrics.

        :return: The date_from of this ReleaseMetricsRequest.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date):
        """Sets the date_from of this ReleaseMetricsRequest.

        Date from when to start measuring the metrics.

        :param date_from: The date_from of this ReleaseMetricsRequest.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this ReleaseMetricsRequest.

        Date up to which to measure the metrics.

        :return: The date_to of this ReleaseMetricsRequest.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date):
        """Sets the date_to of this ReleaseMetricsRequest.

        Date up to which to measure the metrics.

        :param date_to: The date_to of this ReleaseMetricsRequest.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def granularities(self) -> List[str]:
        """Gets the granularities of this ReleaseMetricsRequest.

        :return: The granularities of this ReleaseMetricsRequest.
        """
        return self._granularities

    @granularities.setter
    def granularities(self, granularities: List[str]):
        """Sets the granularities of this ReleaseMetricsRequest.

        :param granularities: The granularities of this ReleaseMetricsRequest.
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
        """Gets the quantiles of this ReleaseMetricsRequest.

        :return: The quantiles of this ReleaseMetricsRequest.
        """
        return self._quantiles

    @quantiles.setter
    def quantiles(self, quantiles: Optional[List[float]]):
        """Sets the quantiles of this ReleaseMetricsRequest.

        :param quantiles: The quantiles of this ReleaseMetricsRequest.
        """
        if quantiles is None:
            self._quantiles = None
            return
        validate_quantiles(quantiles)
        self._quantiles = quantiles

    @property
    def timezone(self) -> int:
        """Gets the timezone of this ReleaseMetricsRequest.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :return: The timezone of this ReleaseMetricsRequest.
        """
        return self._timezone

    @timezone.setter
    def timezone(self, timezone: int):
        """Sets the timezone of this ReleaseMetricsRequest.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :param timezone: The timezone of this ReleaseMetricsRequest.
        """
        if timezone is not None and timezone > 720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value less than or equal to `720`")
        if timezone is not None and timezone < -720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value greater than or equal to `-720`")

        self._timezone = timezone

    @property
    def account(self) -> int:
        """Gets the account of this ReleaseMetricsRequest.

        Session account ID.

        :return: The account of this ReleaseMetricsRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this ReleaseMetricsRequest.

        Session account ID.

        :param account: The account of this ReleaseMetricsRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account
