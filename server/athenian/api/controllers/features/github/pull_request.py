from datetime import datetime
from typing import Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, Type

import networkx as nx
import numpy as np

from athenian.api.controllers.features.histogram import calculate_histogram, Histogram, Scale
from athenian.api.controllers.features.metric import Metric, T
from athenian.api.controllers.features.statistics import mean_confidence_interval, \
    median_confidence_interval
from athenian.api.controllers.miners.types import PullRequestFacts


class PullRequestMetricCalculator(Generic[T]):
    """
    Pull request metric calculator, base abstract class.

    Each call to `PullRequestMetricCalculator()` feeds another PR to update the state.
    `PullRequestMetricCalculator.value()` returns the current metric value.
    `PullRequestMetricCalculator._analyze()` is to be implemented by the particular metric
    calculators.
    """

    # Whether the calc should care about PRs without events on the time interval.
    requires_full_span = False

    # Types of dependencies - upstream PullRequestMetricCalculator-s.
    deps = tuple()

    def __init__(self, *deps: "PullRequestMetricCalculator", quantiles: Sequence[float]):
        """Initialize a new `PullRequestMetricCalculator` instance."""
        self.samples = []
        self._peek = None
        self._last_value = None
        self._calcs = []
        self._quantiles = quantiles
        for calc in deps:
            for cls in self.deps:
                if isinstance(calc, cls):
                    self._calcs.append(calc)

    def __call__(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> bool:
        """Supply another pull request facts to update the state.

        :param facts: Mined facts about another pull request: timestamps, stats, etc.
        :param min_time: Start of the considered time interval. It is needed to discard samples \
                         with both ends less than the minimum time.
        :param max_time: Finish of the considered time interval. It is needed to discard samples \
                         with both ends greater than the maximum time.
        :return: Boolean indicating whether the calculated value exists.
        """
        self._peek = self._analyze(facts, min_time, max_time, **kwargs)
        exists = self._peek is not None
        if exists:
            self.samples.append(self._peek)
        return exists

    def reset(self):
        """Reset the internal state."""
        self.samples.clear()
        self._last_value = None
        for calc in self._calcs:
            calc.reset()

    @property
    def value(self) -> Metric[T]:
        """Return the current metric value."""
        if self._last_value is None:
            self._last_value = self._value(self._cut_by_quantiles())
        return self._last_value

    @property
    def peek(self) -> T:
        """Return the last calculated sample."""
        return self._peek

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[T]:
        """Calculate the actual state update."""
        raise NotImplementedError

    def _value(self, samples: Sequence[T]) -> Metric[T]:
        """Calculate the actual current metric value."""
        raise NotImplementedError

    def _cut_by_quantiles(self) -> Sequence[T]:
        """Cut from the left and the right of the distribution by quantiles."""
        if not self.samples or (self._quantiles[0] == 0 and self._quantiles[1] == 1):
            return self.samples
        cut_values = np.quantile(self.samples, self._quantiles)
        samples = np.asarray(self.samples)
        if self._quantiles[0] != 0:
            samples = np.delete(samples, np.where(samples < cut_values[0])[0])
        samples = np.delete(samples, np.where(samples > cut_values[1])[0])
        return samples


class PullRequestAverageMetricCalculator(PullRequestMetricCalculator[T]):
    """Mean calculator."""

    may_have_negative_values: bool

    def _value(self, samples: Sequence[T]) -> Metric[T]:
        """Calculate the current metric value."""
        if len(samples) == 0:
            return Metric(False, None, None, None)
        assert self.may_have_negative_values is not None
        if not self.may_have_negative_values:
            zero = type(samples[0])(0)
            assert all(s >= zero for s in samples), str(samples)
        return Metric(True, *mean_confidence_interval(samples, self.may_have_negative_values))

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[T]:
        """Calculate the actual state update."""
        raise NotImplementedError


class PullRequestMedianMetricCalculator(PullRequestMetricCalculator[T]):
    """Median calculator."""

    def _value(self, samples: Sequence[T]) -> Metric[T]:
        """Calculate the current metric value."""
        if len(samples) == 0:
            return Metric(False, None, None, None)
        return Metric(True, *median_confidence_interval(samples))

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[T]:
        """Calculate the actual state update."""
        raise NotImplementedError


class PullRequestSumMetricCalculator(PullRequestMetricCalculator[T]):
    """Sum calculator."""

    def _value(self, samples: Sequence[T]) -> Metric[T]:
        """Calculate the current metric value."""
        exists = len(samples) > 0
        return Metric(exists, sum(samples) if exists else None, None, None)

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[T]:
        """Calculate the actual state update."""
        raise NotImplementedError


class PullRequestCounter(PullRequestSumMetricCalculator[int]):
    """Count the number of PRs that were used to calculate the specified metric."""

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[int]:
        """Calculate the actual state update."""
        return int(self._calcs[0].peek is not None)


class PullRequestHistogramCalculator(PullRequestMetricCalculator):
    """Pull request histogram calculator, base abstract class."""

    def histogram(self, scale: Scale, bins: int) -> Histogram[T]:
        """Calculate the histogram over the current distribution."""
        samples = self._cut_by_quantiles()
        if scale == Scale.LOG:
            shift_log = getattr(self, "_shift_log", None)  # type: Optional[Callable[[T], T]]
            if shift_log is not None:
                samples = [shift_log(s) for s in self.samples]
        return calculate_histogram(samples, scale, bins)


metric_calculators: Dict[str, Type[PullRequestMetricCalculator]] = {}
histogram_calculators: Dict[str, Type[PullRequestHistogramCalculator]] = {}


def register_metric(name: str):
    """Keep track of the PR metric calculators and generate the histogram calculator."""
    assert isinstance(name, str)

    def register_with_name(cls: Type[PullRequestMetricCalculator]):
        metric_calculators[name] = cls
        if not issubclass(cls, PullRequestSumMetricCalculator):
            histogram_calculators[name] = \
                type("HistogramOf" + cls.__name__, (cls, PullRequestHistogramCalculator), {})
        return cls

    return register_with_name


class PullRequestMetricCalculatorEnsemble(Generic[T]):
    """Several calculators ordered in sequence such that the dependencies are respected and each \
    calculator class is calculated only once."""

    def __init__(self,
                 *metrics: str,
                 class_mapping: Dict[str, Type[PullRequestMetricCalculator]] = metric_calculators,
                 quantiles: Sequence[float]):
        """Initialize a new instance of PullRequestMetricCalculatorEnsemble class."""
        metric_classes = {class_mapping[m]: m for m in metrics}
        self._calcs, self._metrics = self._plan_classes(metric_classes, quantiles)

    @staticmethod
    def _plan_classes(metric_classes: Dict[Type[PullRequestMetricCalculator], str],
                      quantiles: Sequence[float],
                      ) -> Tuple[List[PullRequestMetricCalculator],
                                 Dict[str, PullRequestMetricCalculator]]:
        dig = nx.DiGraph()
        required_classes = list(metric_classes)
        while required_classes:
            cls = required_classes.pop()
            if cls.deps:
                for dep in cls.deps:
                    if dep not in dig:
                        required_classes.append(dep)
                    dig.add_edge(cls, dep)
            elif cls not in dig:
                dig.add_node(cls)
        calcs = []
        metrics = {}
        cls_instances = {}
        for cls in reversed(list(nx.topological_sort(dig))):
            calc = cls(*[cls_instances[dep] for dep in cls.deps],
                       quantiles=quantiles)
            calcs.append(calc)
            cls_instances[cls] = calc
            try:
                metrics[metric_classes[cls]] = calc
            except KeyError:
                continue
        return calcs, metrics

    def __call__(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> None:
        """Consume another pull request by the owned calculators, changing their states."""
        for calc in self._calcs:
            calc(facts, min_time, max_time, **kwargs)

    def __bool__(self) -> bool:
        """Return True if there is at leat one calculator inside; otherwise, False."""
        return bool(len(self._calcs))

    def values(self) -> Dict[str, Metric[T]]:
        """Calculate the current metric values."""
        return {k: v.value for k, v in self._metrics.items()}

    def reset(self) -> None:
        """Reset the internal states of all the owned calculators."""
        for calc in self._calcs:
            calc.reset()


class PullRequestHistogramCalculatorEnsemble(PullRequestMetricCalculatorEnsemble[T]):
    """Like PullRequestMetricCalculatorEnsemble, but for histograms."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of PullRequestHistogramCalculatorEnsemble class."""
        super().__init__(*metrics, class_mapping=histogram_calculators, quantiles=quantiles)

    def histograms(self, scale: Scale, bins: int) -> Dict[str, Histogram[T]]:
        """Calculate the current histograms."""
        return {k: v.histogram(scale, bins) for k, v in self._metrics.items()}


class BinnedPullRequestMetricCalculator(Generic[T]):
    """Batched metrics calculation on sequential time intervals."""

    def __init__(self,
                 metrics: Sequence[str],
                 time_intervals: Sequence[datetime],
                 quantiles: Sequence[float]):
        """
        Initialize a new instance of `BinnedPullRequestMetricCalculator`.

        :param metrics: Sequence of metric names to calculate in each bin.
        :param time_intervals: Time interval borders in UTC. Each interval spans \
                               `[time_intervals[i], time_intervals[i + 1]]`, the ending \
                               not included.
        """
        self.calcs_regular = PullRequestMetricCalculatorEnsemble(
            *[m for m in metrics if not metric_calculators[m].requires_full_span],
            quantiles=quantiles,
        )
        self.calcs_full_span = PullRequestMetricCalculatorEnsemble(
            *[m for m in metrics if metric_calculators[m].requires_full_span],
            quantiles=quantiles,
        )
        self.metrics = metrics
        assert len(time_intervals) >= 2
        self.time_intervals = time_intervals

    def __call__(self, items: Iterable[PullRequestFacts]) -> List[List[Metric[T]]]:
        """
        Calculate the binned metrics on a series of PullRequestFacts.

        For each time interval we collect the list of PRs that are relevant and measure \
        the specified metrics.
        """
        borders = self.time_intervals
        calcs_regular = self.calcs_regular
        calcs_full_span = self.calcs_full_span
        dummy_bins = [[] for _ in borders[:-1]]
        # avoid multiple evaluations of work_began, it is really visible in the profile
        work_begans = [(item.work_began, i) for i, item in enumerate(items)]
        if not hasattr(items, "__getitem__"):
            items = list(items)
        items = [items[i] for _, i in sorted(work_begans)]
        regular_bins = self._bin_regulars(items) if calcs_regular else dummy_bins
        full_span_bins = self._bin_full_spans(items) if calcs_full_span else dummy_bins
        result = []
        for regular_bin, full_span_bin, time_from, time_to in zip(
                regular_bins, full_span_bins, borders, borders[1:]):
            for item in regular_bin:
                calcs_regular(item, time_from, time_to)
            for item in full_span_bin:
                calcs_full_span(item, time_from, time_to)
            values_dict = calcs_regular.values()
            values_dict.update(calcs_full_span.values())
            result.append([values_dict[m] for m in self.metrics])
            calcs_regular.reset()
            calcs_full_span.reset()
        return result

    def _bin_regulars(self, items: Iterable[PullRequestFacts]) -> List[List[PullRequestFacts]]:
        borders = self.time_intervals
        bins = [[] for _ in borders[:-1]]
        pos = 0
        for item in items:
            while pos < len(borders) - 1 and item.work_began.best > borders[pos + 1]:
                pos += 1
            endpoint = item.max_timestamp()
            span = pos
            while span < len(bins) and endpoint > borders[span]:
                bins[span].append(item)
                span += 1
        return bins

    def _bin_full_spans(self, items: Iterable[PullRequestFacts]) -> List[List[PullRequestFacts]]:
        borders = self.time_intervals
        bins = [[] for _ in borders[:-1]]
        pos = 0
        for item in items:
            while pos < len(borders) - 1 and item.work_began.best > borders[pos + 1]:
                pos += 1
            endpoint = item.released.best if item.released else borders[-1]
            span = pos
            while span < len(bins) and endpoint > borders[span]:
                bins[span].append(item)
                span += 1
        return bins
