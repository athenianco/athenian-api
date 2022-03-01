from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.code_check_metric_id import CodeCheckMetricID
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import CommonMetricsProperties
from athenian.api.models.web.for_set_code_checks import ForSetCodeChecks


class _CodeCheckMetricsRequest(Model, sealed=False):
    """Request for calculating metrics on top of code check runs (CI) data."""

    openapi_types = {
        "for_": List[ForSetCodeChecks],
        "metrics": List[CodeCheckMetricID],
        "split_by_check_runs": Optional[bool],
    }

    attribute_map = {
        "for_": "for",
        "metrics": "metrics",
        "split_by_check_runs": "split_by_check_runs",
    }

    def __init__(
        self,
        for_: List[ForSetCodeChecks] = None,
        metrics: List[CodeCheckMetricID] = None,
        split_by_check_runs: bool = None,
    ):
        """CodeCheckMetricsRequest - a model defined in OpenAPI

        :param for_: The for of this CodeCheckMetricsRequest.
        :param metrics: The metrics of this CodeCheckMetricsRequest.
        :param split_by_check_runs: The split_by_check_runs of this CodeCheckMetricsRequest.
        """
        self._for_ = for_
        self._metrics = metrics
        self._split_by_check_runs = split_by_check_runs

    @property
    def for_(self) -> List[ForSetCodeChecks]:
        """Gets the for_ of this CodeCheckMetricsRequest.

        Sets of developers and repositories for which to calculate the metrics. The aggregation is
        `AND` between repositories and developers. The aggregation is `OR` inside both repositories
        and developers.

        :return: The for_ of this CodeCheckMetricsRequest.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: List[ForSetCodeChecks]):
        """Sets the for_ of this CodeCheckMetricsRequest.

        Sets of developers and repositories for which to calculate the metrics. The aggregation is
        `AND` between repositories and developers. The aggregation is `OR` inside both repositories
        and developers.

        :param for_: The for_ of this CodeCheckMetricsRequest.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for_`, must not be `None`")

        self._for_ = for_

    @property
    def metrics(self) -> List[CodeCheckMetricID]:
        """Gets the metrics of this CodeCheckMetricsRequest.

        Requested metric identifiers.

        :return: The metrics of this CodeCheckMetricsRequest.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[CodeCheckMetricID]):
        """Sets the metrics of this CodeCheckMetricsRequest.

        Requested metric identifiers.

        :param metrics: The metrics of this CodeCheckMetricsRequest.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")

        self._metrics = metrics

    @property
    def split_by_check_runs(self) -> Optional[bool]:
        """Gets the split_by_check_runs of this CodeCheckMetricsRequest.

        Calculate metrics separately for each number of check runs in suite.

        :return: The split_by_check_runs of this CodeCheckMetricsRequest.
        """
        return self._split_by_check_runs

    @split_by_check_runs.setter
    def split_by_check_runs(self, split_by_check_runs: Optional[bool]):
        """Sets the split_by_check_runs of this CodeCheckMetricsRequest.

        Calculate metrics separately for each number of check runs in suite.

        :param split_by_check_runs: The split_by_check_runs of this CodeCheckMetricsRequest.
        """
        self._split_by_check_runs = split_by_check_runs


CodeCheckMetricsRequest = AllOf(_CodeCheckMetricsRequest,
                                CommonFilterProperties,
                                CommonMetricsProperties,
                                name="CodeCheckMetricsRequest",
                                module=__name__)
