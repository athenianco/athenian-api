from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import GranularitiesMixin
from athenian.api.models.web.developer_metric_id import DeveloperMetricID
from athenian.api.models.web.for_set_developers import ForSetDevelopers


class _DeveloperMetricsRequest(Model, GranularitiesMixin, sealed=False):
    """Request for calculating metrics on developer activities."""

    for_: (list[ForSetDevelopers], "for")
    metrics: list[str]
    granularities: list[str]

    def validate_metrics(self, metrics: list[str]) -> list[str]:
        """Sets the metrics of this DeveloperMetricsRequest.

        Requested metric identifiers.

        :param metrics: The metrics of this DeveloperMetricsRequest.
        """
        if metrics is None:
            raise ValueError("Invalid value for `metrics`, must not be `None`")
        diff = set(metrics) - set(DeveloperMetricID)
        if diff:
            raise ValueError("Unsupported values of `metrics`: %s" % diff)

        return metrics


DeveloperMetricsRequest = AllOf(
    _DeveloperMetricsRequest,
    CommonFilterProperties,
    name="DeveloperMetricsRequest",
    module=__name__,
)
