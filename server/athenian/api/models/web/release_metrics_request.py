from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.granularity import Granularity
from athenian.api.models.web.jira_filter import JIRAFilter
from athenian.api.models.web.quantiles import validate_quantiles
from athenian.api.models.web.release_metric_id import ReleaseMetricID
from athenian.api.models.web.release_with import ReleaseWith


class _ReleaseMetricsRequest(Model):
    """Request of `/metrics/releases` to calculate metrics on releases."""

    openapi_types = {
        "for_": List[List[str]],
        "with_": Optional[List[ReleaseWith]],
        "metrics": List[str],
        "granularities": List[str],
        "quantiles": List[float],
        "jira": Optional[JIRAFilter],
    }

    attribute_map = {
        "for_": "for",
        "with_": "with",
        "metrics": "metrics",
        "granularities": "granularities",
        "quantiles": "quantiles",
        "jira": "jira",
    }

    __enable_slots__ = False

    def __init__(
        self,
        for_: Optional[List[List[str]]] = None,
        with_: Optional[List[ReleaseWith]] = None,
        metrics: Optional[List[str]] = None,
        granularities: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        jira: Optional[JIRAFilter] = None,
    ):
        """ReleaseMetricsRequest - a model defined in OpenAPI

        :param for_: The for of this ReleaseMetricsRequest.
        :param with_: The with of this ReleaseMetricsRequest.
        :param metrics: The metrics of this ReleaseMetricsRequest.
        :param granularities: The granularities of this ReleaseMetricsRequest.
        :param jira: The jira of this ReleaseMetricsRequest.
        """
        self._for_ = for_
        self._with_ = with_
        self._metrics = metrics
        self._granularities = granularities
        self._quantiles = quantiles
        self._jira = jira

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
                raise ValueError("`metrics[%d]` = '%s' must be one of %s" % (
                    i, metric, list(ReleaseMetricID)))

        self._metrics = metrics

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
    def jira(self) -> Optional[JIRAFilter]:
        """Gets the jira of this ReleaseMetricsRequest.

        :return: The jira of this ReleaseMetricsRequest.
        """
        return self._jira

    @jira.setter
    def jira(self, jira: Optional[JIRAFilter]):
        """Sets the jira of this ReleaseMetricsRequest.

        :param jira: The jira of this ReleaseMetricsRequest.
        """
        self._jira = jira


ReleaseMetricsRequest = AllOf(_ReleaseMetricsRequest, CommonFilterProperties,
                              name="ReleaseMetricsRequest", module=__name__)
