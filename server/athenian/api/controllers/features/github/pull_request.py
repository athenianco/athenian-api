from datetime import datetime
from typing import Dict, Sequence, Type

from athenian.api.controllers.features.metric_calculator import BinnedMetricCalculator, \
    HistogramCalculator, \
    HistogramCalculatorEnsemble, MetricCalculator, MetricCalculatorEnsemble, SumMetricCalculator
from athenian.api.controllers.miners.types import T

metric_calculators: Dict[str, Type[MetricCalculator]] = {}
histogram_calculators: Dict[str, Type[HistogramCalculator]] = {}


def register_metric(name: str):
    """Keep track of the PR metric calculators and generate the histogram calculator."""
    assert isinstance(name, str)

    def register_with_name(cls: Type[MetricCalculator]):
        metric_calculators[name] = cls
        if not issubclass(cls, SumMetricCalculator):
            histogram_calculators[name] = \
                type("HistogramOf" + cls.__name__, (cls, HistogramCalculator), {})
        return cls

    return register_with_name


class PullRequestMetricCalculatorEnsemble(MetricCalculatorEnsemble[T]):
    """MetricCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of PullRequestMetricCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=metric_calculators)


class PullRequestHistogramCalculatorEnsemble(HistogramCalculatorEnsemble[T]):
    """HistogramCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of PullRequestHistogramCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=histogram_calculators)


class PullRequestBinnedMetricCalculator(BinnedMetricCalculator[T]):
    """BinnedMetricCalculator adapted for pull requests."""

    def __init__(self,
                 metrics: Sequence[str],
                 time_intervals: Sequence[datetime],
                 quantiles: Sequence[float]):
        """Initialize a new instance of PullRequestBinnedMetricCalculator class."""
        super().__init__(metrics=metrics, time_intervals=time_intervals, quantiles=quantiles,
                         class_mapping=metric_calculators,
                         start_time_getter=lambda pr: pr.work_began,
                         finish_time_getter=lambda pr: pr.released.best)
