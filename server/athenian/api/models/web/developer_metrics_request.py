from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.common_filter_properties import CommonFilterPropertiesMixin
from athenian.api.models.web.developer_metric_id import DeveloperMetricID
from athenian.api.models.web.for_set_developers import ForSetDevelopers
from athenian.api.models.web.granularity import Granularity


class DeveloperMetricsRequest(Model, CommonFilterPropertiesMixin):
    """Request for calculating metrics on top of developer activities."""

    openapi_types = {
        "for_": List[ForSetDevelopers],
        "metrics": List[str],
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "account": int,
        "granularities": List[str],
    }

    attribute_map = {
        "for_": "for",
        "metrics": "metrics",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "account": "account",
        "granularities": "granularities",
    }

    def __init__(
        self,
        for_: Optional[List[ForSetDevelopers]] = None,
        metrics: Optional[List[str]] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        account: Optional[int] = None,
        granularities: Optional[List[str]] = None,
    ):
        """DeveloperMetricsRequest - a model defined in OpenAPI

        :param for_: The for_ of this DeveloperMetricsRequest.
        :param metrics: The metrics of this DeveloperMetricsRequest.
        :param date_from: The date_from of this DeveloperMetricsRequest.
        :param date_to: The date_to of this DeveloperMetricsRequest.
        :param timezone: The timezone of this DeveloperMetricsRequest.
        :param account: The account of this DeveloperMetricsRequest.
        :param granularities: The granularities of this DeveloperMetricsRequest.
        """
        self._for_ = for_
        self._metrics = metrics
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._account = account
        self._granularities = granularities

    @property
    def for_(self) -> List[ForSetDevelopers]:
        """Gets the for_ of this DeveloperMetricsRequest.

        Sets of developers and repositories to calculate the metrics for.

        :return: The for_ of this DeveloperMetricsRequest.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: List[ForSetDevelopers]):
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
        diff = set(metrics) - set(DeveloperMetricID)
        if diff:
            raise ValueError("Unsupported values of `metrics`: %s" % diff)

        self._metrics = metrics

    @property
    def granularities(self) -> List[str]:
        """Gets the granularities of this DeveloperMetricsRequest.

        Splits of the specified time range `[date_from, date_to)`.

        :return: The granularities of this DeveloperMetricsRequest.
        """
        return self._granularities

    @granularities.setter
    def granularities(self, granularities: List[str]):
        """Sets the granularities of this DeveloperMetricsRequest.

        Splits of the specified time range `[date_from, date_to)`.

        :param granularities: The granularities of this DeveloperMetricsRequest.
        """
        # FIXME(vmarkovtsev): uncomment when the webapp is ready
        # if granularities is None:
        #     raise ValueError("Invalid value for `granularities`, must not be `None`")
        for i, g in enumerate(granularities):
            if not Granularity.format.match(g):
                raise ValueError(
                    'Invalid value for `granularity[%d]`: "%s"` does not match /%s/' %
                    (i, g, Granularity.format.pattern))

        self._granularities = granularities
