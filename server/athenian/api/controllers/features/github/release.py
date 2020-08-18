from datetime import datetime
from typing import Dict, Sequence, Type

from athenian.api.controllers.features.metric_calculator import BinnedMetricCalculator, \
    MetricCalculator, MetricCalculatorEnsemble
from athenian.api.controllers.miners.types import T

metric_calculators: Dict[str, Type[MetricCalculator]] = {}


def register_metric(name: str):
    """Keep track of the release metric calculators."""
    assert isinstance(name, str)

    def register_with_name(cls: Type[MetricCalculator]):
        metric_calculators[name] = cls
        return cls

    return register_with_name


class ReleaseMetricCalculatorEnsemble(MetricCalculatorEnsemble[T]):
    """MetricCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of ReleaseMetricCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=metric_calculators)


class ReleaseBinnedMetricCalculator(BinnedMetricCalculator[T]):
    """BinnedMetricCalculator adapted for pull requests."""

    def __init__(self,
                 metrics: Sequence[str],
                 time_intervals: Sequence[datetime],
                 quantiles: Sequence[float]):
        """Initialize a new instance of ReleaseBinnedMetricCalculator class."""
        super().__init__(metrics=metrics, time_intervals=time_intervals, quantiles=quantiles,
                         class_mapping=metric_calculators,
                         start_time_getter=lambda r: r.published_at,
                         finish_time_getter=lambda r: r.published_at)
