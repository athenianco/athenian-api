from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_developer_metrics_item_deprecated import \
    CalculatedDeveloperMetricsItemDeprecated
from athenian.api.models.web.developer_metric_id import DeveloperMetricID


class CalculatedDeveloperMetricsDeprecated(Model):
    """Response from /metrics/developers - calculated metrics over developer activities."""

    openapi_types = {
        "calculated": List[CalculatedDeveloperMetricsItemDeprecated],
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "metrics": List[DeveloperMetricID],
    }

    attribute_map = {
        "calculated": "calculated",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "metrics": "metrics",
    }

    def __init__(
        self,
        calculated: Optional[List[CalculatedDeveloperMetricsItemDeprecated]] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        metrics: Optional[List[DeveloperMetricID]] = None,
    ):
        """CalculatedDeveloperMetricsDeprecated - a model defined in OpenAPI

        :param calculated: The calculated of this CalculatedDeveloperMetricsDeprecated.
        :param date_from: The date_from of this CalculatedDeveloperMetricsDeprecated.
        :param date_to: The date_to of this CalculatedDeveloperMetricsDeprecated.
        :param timezone: The timezone of this CalculatedDeveloperMetricsDeprecated.
        :param metrics: The metrics of this CalculatedDeveloperMetricsDeprecated.
        """
        self._calculated = calculated
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._metrics = metrics

    @property
    def calculated(self) -> List[CalculatedDeveloperMetricsItemDeprecated]:
        """Gets the calculated of this CalculatedDeveloperMetricsDeprecated.

        Values of the requested metrics on the given time interval.

        :return: The calculated of this CalculatedDeveloperMetricsDeprecated.
        """
        return self._calculated

    @calculated.setter
    def calculated(self, calculated: List[CalculatedDeveloperMetricsItemDeprecated]):
        """Sets the calculated of this CalculatedDeveloperMetricsDeprecated.

        Values of the requested metrics on the given time interval.

        :param calculated: The calculated of this CalculatedDeveloperMetricsDeprecated.
        """
        if calculated is None:
            raise ValueError("Invalid value for `calculated`, must not be `None`")

        self._calculated = calculated

    @property
    def date_from(self) -> date:
        """Gets the date_from of this CalculatedDeveloperMetricsDeprecated.

        Repeats `DeveloperMetricsRequest.date_from`.

        :return: The date_from of this CalculatedDeveloperMetricsDeprecated.
        """
        return self._date_from

    @date_from.setter
    def date_from(self, date_from: date):
        """Sets the date_from of this CalculatedDeveloperMetricsDeprecated.

        Repeats `DeveloperMetricsRequest.date_from`.

        :param date_from: The date_from of this CalculatedDeveloperMetricsDeprecated.
        """
        if date_from is None:
            raise ValueError("Invalid value for `date_from`, must not be `None`")

        self._date_from = date_from

    @property
    def date_to(self) -> date:
        """Gets the date_to of this CalculatedDeveloperMetricsDeprecated.

        Repeats `DeveloperMetricsRequest.date_to`.

        :return: The date_to of this CalculatedDeveloperMetricsDeprecated.
        """
        return self._date_to

    @date_to.setter
    def date_to(self, date_to: date):
        """Sets the date_to of this CalculatedDeveloperMetricsDeprecated.

        Repeats `DeveloperMetricsRequest.date_to`.

        :param date_to: The date_to of this CalculatedDeveloperMetricsDeprecated.
        """
        if date_to is None:
            raise ValueError("Invalid value for `date_to`, must not be `None`")

        self._date_to = date_to

    @property
    def timezone(self) -> int:
        """Gets the timezone of this CalculatedDeveloperMetricsDeprecated.

        Repeats `DeveloperMetricsRequest.timezone`.

        :return: The timezone of this CalculatedDeveloperMetricsDeprecated.
        """
        return self._timezone

    @timezone.setter
    def timezone(self, timezone: int):
        """Sets the timezone of this CalculatedDeveloperMetricsDeprecated.

        Repeats `DeveloperMetricsRequest.timezone`.

        :param timezone: The timezone of this CalculatedDeveloperMetricsDeprecated.
        """
        if timezone is not None and timezone > 720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value less than or equal to `720`")
        if timezone is not None and timezone < -720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value greater than or equal to `-720`")

        self._timezone = timezone

    @property
    def metrics(self) -> List[DeveloperMetricID]:
        """Gets the metrics of this CalculatedDeveloperMetricsDeprecated.

        Repeats `DeveloperMetricsRequest.metrics`.

        :return: The metrics of this CalculatedDeveloperMetricsDeprecated.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[DeveloperMetricID]):
        """Sets the metrics of this CalculatedDeveloperMetricsDeprecated.

        Repeats `DeveloperMetricsRequest.metrics`.

        :param metrics: The metrics of this CalculatedDeveloperMetricsDeprecated.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")

        self._metrics = metrics
