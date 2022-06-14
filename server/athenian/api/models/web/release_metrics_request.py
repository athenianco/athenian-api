from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import CommonMetricsProperties
from athenian.api.models.web.for_set_common import CommonPullRequestFilters
from athenian.api.models.web.release_metric_id import ReleaseMetricID
from athenian.api.models.web.release_with import ReleaseWith


class _ReleaseMetricsRequest(Model, sealed=False):
    """Request of `/metrics/releases` to calculate metrics on releases."""

    attribute_types = {
        "for_": List[List[str]],
        "with_": Optional[List[ReleaseWith]],
        "metrics": List[str],
    }

    attribute_map = {
        "for_": "for",
        "with_": "with",
        "metrics": "metrics",
    }

    def __init__(
        self,
        for_: Optional[List[List[str]]] = None,
        with_: Optional[List[ReleaseWith]] = None,
        metrics: Optional[List[str]] = None,
    ):
        """ReleaseMetricsRequest - a model defined in OpenAPI

        :param for_: The for of this ReleaseMetricsRequest.
        :param with_: The with of this ReleaseMetricsRequest.
        :param metrics: The metrics of this ReleaseMetricsRequest.
        """
        self._for_ = for_
        self._with_ = with_
        self._metrics = metrics

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
    def with_(self) -> Optional[List[ReleaseWith]]:
        """Gets the with_ of this ReleaseMetricsRequest.

        Release contribution roles.

        :return: The with_ of this ReleaseMetricsRequest.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[List[ReleaseWith]]):
        """Sets the with_ of this ReleaseMetricsRequest.

        Release contribution roles.

        :param with_: The with_ of this ReleaseMetricsRequest.
        """
        self._with_ = with_

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
                raise ValueError(
                    "`metrics[%d]` = '%s' must be one of %s" % (i, metric, list(ReleaseMetricID))
                )

        self._metrics = metrics


ReleaseMetricsRequest = AllOf(
    _ReleaseMetricsRequest,
    CommonFilterProperties,
    CommonMetricsProperties,
    CommonPullRequestFilters,
    name="ReleaseMetricsRequest",
    module=__name__,
)
