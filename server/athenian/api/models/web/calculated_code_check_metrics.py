from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_code_check_metrics_item import (
    CalculatedCodeCheckMetricsItem,
)
from athenian.api.models.web.code_check_metric_id import CodeCheckMetricID


class CalculatedCodeCheckMetrics(Model):
    """Response from `/metrics/code_checks`."""

    attribute_types = {
        "calculated": List[CalculatedCodeCheckMetricsItem],
        "metrics": List[CodeCheckMetricID],
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "granularities": List[str],
        "quantiles": Optional[List[float]],
        "split_by_check_runs": bool,
    }

    attribute_map = {
        "calculated": "calculated",
        "metrics": "metrics",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "granularities": "granularities",
        "quantiles": "quantiles",
        "split_by_check_runs": "split_by_check_runs",
    }

    def __init__(
        self,
        calculated: Optional[List[CalculatedCodeCheckMetricsItem]] = None,
        metrics: Optional[List[CodeCheckMetricID]] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        granularities: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        split_by_check_runs: Optional[bool] = None,
    ):
        """CalculatedCodeCheckMetrics - a model defined in OpenAPI

        :param calculated: The calculated of this CalculatedCodeCheckMetrics.
        :param metrics: The metrics of this CalculatedCodeCheckMetrics.
        :param date_from: The date_from of this CalculatedCodeCheckMetrics.
        :param date_to: The date_to of this CalculatedCodeCheckMetrics.
        :param timezone: The timezone of this CalculatedCodeCheckMetrics.
        :param granularities: The granularities of this CalculatedCodeCheckMetrics.
        :param quantiles: The quantiles of this CalculatedCodeCheckMetrics.
        :param split_by_check_runs: The split_by_check_runs of this CalculatedCodeCheckMetrics.
        """
        self._calculated = calculated
        self._metrics = metrics
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._granularities = granularities
        self._quantiles = quantiles
        self._split_by_check_runs = split_by_check_runs

    @property
    def calculated(self) -> List[CalculatedCodeCheckMetricsItem]:
        """Gets the calculated of this CalculatedCodeCheckMetrics.

        Values of the requested metrics through time.

        :return: The calculated of this CalculatedCodeCheckMetrics.
        """
        return self._calculated

    @calculated.setter
    def calculated(self, calculated: List[CalculatedCodeCheckMetricsItem]):
        """Sets the calculated of this CalculatedCodeCheckMetrics.

        Values of the requested metrics through time.

        :param calculated: The calculated of this CalculatedCodeCheckMetrics.
        """
        if calculated is None:
            raise ValueError("Invalid value for `calculated`, must not be `None`")

        self._calculated = calculated

    @property
    def metrics(self) -> List[CodeCheckMetricID]:
        """Gets the metrics of this CalculatedCodeCheckMetrics.

        Repeats `CodeCheckMetricsRequest.metrics`.

        :return: The metrics of this CalculatedCodeCheckMetrics.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[CodeCheckMetricID]):
        """Sets the metrics of this CalculatedCodeCheckMetrics.

        Repeats `CodeCheckMetricsRequest.metrics`.

        :param metrics: The metrics of this CalculatedCodeCheckMetrics.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")

        self._metrics = metrics

    @property
    def date_from(self) -> date:
        """Gets the date_from of this CalculatedCodeCheckMetrics.

        Repeats `CodeCheckMetricsRequest.date_from`.

        :return: The date_from of this CalculatedCodeCheckMetrics.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date):
        """Sets the date_from of this CalculatedCodeCheckMetrics.

        Repeats `CodeCheckMetricsRequest.date_from`.

        :param date_from: The date_from of this CalculatedCodeCheckMetrics.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this CalculatedCodeCheckMetrics.

        Repeats `CodeCheckMetricsRequest.date_to`.

        :return: The date_to of this CalculatedCodeCheckMetrics.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date):
        """Sets the date_to of this CalculatedCodeCheckMetrics.

        Repeats `CodeCheckMetricsRequest.date_to`.

        :param date_to: The date_to of this CalculatedCodeCheckMetrics.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def timezone(self) -> int:
        """Gets the timezone of this CalculatedCodeCheckMetrics.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :return: The timezone of this CalculatedCodeCheckMetrics.
        """
        return self._timezone

    @timezone.setter
    def timezone(self, timezone: int):
        """Sets the timezone of this CalculatedCodeCheckMetrics.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :param timezone: The timezone of this CalculatedCodeCheckMetrics.
        """
        if timezone is not None and timezone > 720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value less than or equal to `720`"
            )
        if timezone is not None and timezone < -720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value greater than or equal to `-720`"
            )

        self._timezone = timezone

    @property
    def granularities(self) -> List[str]:
        """Gets the granularities of this CalculatedCodeCheckMetrics.

        :return: The granularities of this CalculatedCodeCheckMetrics.
        """
        return self._granularities

    @granularities.setter
    def granularities(self, granularities: List[str]):
        """Sets the granularities of this CalculatedCodeCheckMetrics.

        :param granularities: The granularities of this CalculatedCodeCheckMetrics.
        """
        if granularities is None:
            raise ValueError("Invalid value for `granularities`, must not be `None`")

        self._granularities = granularities

    @property
    def quantiles(self) -> List[float]:
        """Gets the quantiles of this CalculatedCodeCheckMetrics.

        Cut the distributions at certain quantiles. The default values are [0, 1] which means
        no cutting.

        :return: The quantiles of this CalculatedCodeCheckMetrics.
        """
        return self._quantiles

    @quantiles.setter
    def quantiles(self, quantiles: List[float]):
        """Sets the quantiles of this CalculatedCodeCheckMetrics.

        Cut the distributions at certain quantiles. The default values are [0, 1] which means
        no cutting.

        :param quantiles: The quantiles of this CalculatedCodeCheckMetrics.
        """
        self._quantiles = quantiles

    @property
    def split_by_check_runs(self) -> bool:
        """Gets the split_by_check_runs of this CalculatedCodeCheckMetrics.

        Repeats `CodeCheckMetricsRequest.split_by_check_runs`.

        :return: The split_by_check_runs of this CalculatedCodeCheckMetrics.
        """
        return self._split_by_check_runs

    @split_by_check_runs.setter
    def split_by_check_runs(self, split_by_check_runs: bool):
        """Sets the split_by_check_runs of this CalculatedCodeCheckMetrics.

        Repeats `CodeCheckMetricsRequest.split_by_check_runs`.

        :param split_by_check_runs: The split_by_check_runs of this CalculatedCodeCheckMetrics.
        """
        self._split_by_check_runs = split_by_check_runs
