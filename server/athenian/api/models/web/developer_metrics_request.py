from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import GranularitiesMixin
from athenian.api.models.web.developer_metric_id import DeveloperMetricID
from athenian.api.models.web.for_set_developers import ForSetDevelopers


class _DeveloperMetricsRequest(Model, GranularitiesMixin, sealed=False):
    """Request for calculating metrics on developer activities."""

    attribute_types = {
        "for_": List[ForSetDevelopers],
        "metrics": List[str],
        "granularities": List[str],
    }

    attribute_map = {
        "for_": "for",
        "metrics": "metrics",
        "granularities": "granularities",
    }

    def __init__(
        self,
        for_: Optional[List[ForSetDevelopers]] = None,
        metrics: Optional[List[str]] = None,
        granularities: Optional[List[str]] = None,
    ):
        """DeveloperMetricsRequest - a model defined in OpenAPI

        :param for_: The for_ of this DeveloperMetricsRequest.
        :param metrics: The metrics of this DeveloperMetricsRequest.
        :param granularities: The granularities of this DeveloperMetricsRequest.
        """
        self._for_ = for_
        self._metrics = metrics
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


DeveloperMetricsRequest = AllOf(
    _DeveloperMetricsRequest,
    CommonFilterProperties,
    name="DeveloperMetricsRequest",
    module=__name__,
)
