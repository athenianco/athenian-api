from typing import Dict, List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import CalculatedLinearMetricValues
from athenian.api.models.web.granularity import GranularityMixin
from athenian.api.models.web.release_metric_id import ReleaseMetricID
from athenian.api.models.web.release_with import ReleaseWith


class CalculatedReleaseMetric(Model, GranularityMixin):
    """Response from `/metrics/releases`."""

    attribute_types = {
        "for_": List[str],
        "with_": Optional[ReleaseWith],
        "matches": Dict[str, str],
        "metrics": List[str],
        "granularity": str,
        "values": List[CalculatedLinearMetricValues],
    }

    attribute_map = {
        "for_": "for",
        "with_": "with",
        "matches": "matches",
        "metrics": "metrics",
        "granularity": "granularity",
        "values": "values",
    }

    def __init__(
        self,
        for_: List[str] = None,
        with_: Optional[ReleaseWith] = None,
        matches: Optional[Dict[str, str]] = None,
        metrics: List[str] = None,
        granularity: str = None,
        values: List[CalculatedLinearMetricValues] = None,
    ):
        """CalculatedReleaseMetric - a model defined in OpenAPI

        :param for_: The for of this CalculatedReleaseMetric.
        :param with_: The with of this CalculatedReleaseMetric.
        :param matches: The matches of this CalculatedReleaseMetric.
        :param metrics: The metrics of this CalculatedReleaseMetric.
        :param granularity: The granularity of this CalculatedReleaseMetric.
        :param values: The values of this CalculatedReleaseMetric.
        """
        self._for_ = for_
        self._with_ = with_
        self._matches = matches
        self._metrics = metrics
        self._granularity = granularity
        self._values = values

    @property
    def for_(self) -> List[str]:
        """Gets the for_ of this CalculatedReleaseMetric.

        :return: The for_ of this CalculatedReleaseMetric.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: List[str]):
        """Sets the for_ of this CalculatedReleaseMetric.

        :param for_: The for_ of this CalculatedReleaseMetric.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for_`, must not be `None`")

        self._for_ = for_

    @property
    def with_(self) -> Optional[ReleaseWith]:
        """Gets the with_ of this CalculatedReleaseMetric.

        Release contribution roles.

        :return: The with_ of this CalculatedReleaseMetric.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[ReleaseWith]):
        """Sets the with_ of this CalculatedReleaseMetric.

        Release contribution roles.

        :param with_: The with_ of this CalculatedReleaseMetric.
        """
        self._with_ = with_

    @property
    def matches(self) -> Dict[str, str]:
        """Gets the matches of this CalculatedReleaseMetric.

        :return: The matches of this CalculatedReleaseMetric.
        """
        return self._matches

    @matches.setter
    def matches(self, matches: Dict[str, str]):
        """Sets the matches of this CalculatedReleaseMetric.

        :param matches: The matches of this CalculatedReleaseMetric.
        """
        if matches is None:
            raise ValueError("Invalid value for `matches`, must not be `None`")

        self._matches = matches

    @property
    def metrics(self) -> List[str]:
        """Gets the metrics of this CalculatedReleaseMetric.

        :return: The metrics of this CalculatedReleaseMetric.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[str]):
        """Sets the metrics of this CalculatedReleaseMetric.

        :param metrics: The metrics of this CalculatedReleaseMetric.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        for i, metric in enumerate(metrics):
            if metric not in ReleaseMetricID:
                raise ValueError("`metrics[%d]` = '%s' must be one of %s" % (
                    i, metric, list(ReleaseMetricID)))

        self._metrics = metrics

    @property
    def values(self) -> List[CalculatedLinearMetricValues]:
        """Gets the values of this CalculatedReleaseMetric.

        :return: The values of this CalculatedReleaseMetric.
        """
        return self._values

    @values.setter
    def values(self, values: List[CalculatedLinearMetricValues]):
        """Sets the values of this CalculatedReleaseMetric.

        :param values: The values of this CalculatedReleaseMetric.
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values
