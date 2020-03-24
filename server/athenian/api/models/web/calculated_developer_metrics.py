from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_developer_metrics_item import \
    CalculatedDeveloperMetricsItem
from athenian.api.models.web.developer_metric_id import DeveloperMetricID


class CalculatedDeveloperMetrics(Model):
    """Response from /metrics/developers - calculated metrics over developer activities."""

    def __init__(
        self,
        calculated: Optional[List[CalculatedDeveloperMetricsItem]] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        metrics: Optional[List[DeveloperMetricID]] = None,
    ):
        """CalculatedDeveloperMetrics - a model defined in OpenAPI

        :param calculated: The calculated of this CalculatedDeveloperMetrics.
        :param date_from: The date_from of this CalculatedDeveloperMetrics.
        :param date_to: The date_to of this CalculatedDeveloperMetrics.
        :param metrics: The metrics of this CalculatedDeveloperMetrics.
        """
        self.openapi_types = {
            "calculated": List[CalculatedDeveloperMetricsItem],
            "date_from": date,
            "date_to": date,
            "metrics": List[DeveloperMetricID],
        }

        self.attribute_map = {
            "calculated": "calculated",
            "date_from": "date_from",
            "date_to": "date_to",
            "metrics": "metrics",
        }

        self._calculated = calculated
        self._date_from = date_from
        self._date_to = date_to
        self._metrics = metrics

    @property
    def calculated(self):
        """Gets the calculated of this CalculatedDeveloperMetrics.

        Values of the requested metrics on the given time interval.

        :return: The calculated of this CalculatedDeveloperMetrics.
        :rtype: List[CalculatedDeveloperMetricsItem]
        """
        return self._calculated

    @calculated.setter
    def calculated(self, calculated):
        """Sets the calculated of this CalculatedDeveloperMetrics.

        Values of the requested metrics on the given time interval.

        :param calculated: The calculated of this CalculatedDeveloperMetrics.
        :type calculated: List[CalculatedDeveloperMetricsItem]
        """
        if calculated is None:
            raise ValueError("Invalid value for `calculated`, must not be `None`")

        self._calculated = calculated

    @property
    def date_from(self):
        """Gets the date_from of this CalculatedDeveloperMetrics.

        Repeats `DeveloperMetricsRequest.date_from`.

        :return: The date_from of this CalculatedDeveloperMetrics.
        :rtype: date
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from):
        """Sets the date_from of this CalculatedDeveloperMetrics.

        Repeats `DeveloperMetricsRequest.date_from`.

        :param date_from: The date_from of this CalculatedDeveloperMetrics.
        :type date_from: date
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self):
        """Gets the date_to of this CalculatedDeveloperMetrics.

        Repeats `DeveloperMetricsRequest.date_to`.

        :return: The date_to of this CalculatedDeveloperMetrics.
        :rtype: date
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to):
        """Sets the date_to of this CalculatedDeveloperMetrics.

        Repeats `DeveloperMetricsRequest.date_to`.

        :param date_to: The date_to of this CalculatedDeveloperMetrics.
        :type date_to: date
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def metrics(self):
        """Gets the metrics of this CalculatedDeveloperMetrics.

        Repeats `DeveloperMetricsRequest.metrics`.

        :return: The metrics of this CalculatedDeveloperMetrics.
        :rtype: List[DeveloperMetricID]
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        """Sets the metrics of this CalculatedDeveloperMetrics.

        Repeats `DeveloperMetricsRequest.metrics`.

        :param metrics: The metrics of this CalculatedDeveloperMetrics.
        :type metrics: List[DeveloperMetricID]
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")

        self._metrics = metrics
