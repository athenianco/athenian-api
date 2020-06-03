from datetime import datetime
from itertools import chain
from typing import Dict, Generic, Iterable, List, Optional, Sequence, Type

from athenian.api.controllers.features.metric import Metric, T
from athenian.api.controllers.features.statistics import mean_confidence_interval, \
    median_confidence_interval
from athenian.api.controllers.miners.types import PullRequestTimes


class PullRequestMetricCalculator(Generic[T]):
    """
    Pull request metric calculator, base abstract class.

    Each call to `PullRequestMetricCalculator()` feeds another PR to update the state.
    `PullRequestMetricCalculator.value()` returns the current metric value.
    `PullRequestMetricCalculator.analyze()` is to be implemented by the particular metric
    calculators.
    """

    # This indicates whether the calc should care about PRs without events on the time interval.
    requires_full_span = False

    def __init__(self):
        """Initialize a new `PullRequestMetricCalculator` instance."""
        self.samples = []

    def __call__(self, times: PullRequestTimes, min_time: datetime, max_time: datetime) -> bool:
        """Supply another pull request timestamps to update the state.

        :param min_time: Start of the considered time interval. It is needed to discard samples \
                         with both ends less than the minimum time.
        :param max_time: Finish of the considered time interval. It is needed to discard samples \
                         with both ends greater than the maximum time.
        :return: Boolean indicating whether the calculated value exists.
        """
        sample = self.analyze(times, min_time, max_time)
        exists = sample is not None
        if exists:
            self.samples.append(sample)
        return exists

    def reset(self):
        """Reset the internal state."""
        self.samples.clear()

    def value(self) -> Metric[T]:
        """Calculate the current metric value."""
        raise NotImplementedError

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[T]:
        """Calculate the actual state update."""
        raise NotImplementedError


class PullRequestAverageMetricCalculator(PullRequestMetricCalculator[T]):
    """Mean calculator."""

    may_have_negative_values: bool

    def value(self) -> Metric[T]:
        """Calculate the current metric value."""
        if not self.samples:
            return Metric(False, None, None, None)
        assert self.may_have_negative_values is not None
        if not self.may_have_negative_values:
            zero = type(self.samples[0])(0)
            assert all(s >= zero for s in self.samples), str(self.samples)
        return Metric(True, *mean_confidence_interval(self.samples, self.may_have_negative_values))

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[T]:
        """Calculate the actual state update."""
        raise NotImplementedError


class PullRequestMedianMetricCalculator(PullRequestMetricCalculator[T]):
    """Median calculator."""

    def value(self) -> Metric[T]:
        """Calculate the current metric value."""
        if not self.samples:
            return Metric(False, None, None, None)
        return Metric(True, *median_confidence_interval(self.samples))

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[T]:
        """Calculate the actual state update."""
        raise NotImplementedError


class PullRequestSumMetricCalculator(PullRequestMetricCalculator[T]):
    """Sum calculator."""

    def value(self) -> Metric[T]:
        """Calculate the current metric value."""
        exists = bool(self.samples)
        return Metric(exists, sum(self.samples) if exists else None, None, None)

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[T]:
        """Calculate the actual state update."""
        raise NotImplementedError


class PullRequestCounter(PullRequestSumMetricCalculator[int]):
    """Count the number of PRs that were used to calculate the specified metric."""

    calc_cls: Type[PullRequestMetricCalculator[T]]  # Metric class

    def __init__(self):
        """Initialize a new instance of PullRequestCounter."""
        super().__init__()
        self.calc = self.calc_cls()

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[int]:
        """Calculate the actual state update."""
        return int(self.calc.analyze(times, min_time, max_time) is not None)


calculators: Dict[str, Type[PullRequestMetricCalculator]] = {}


def register(name: str):
    """Keep track of the PR metric calculators."""
    assert isinstance(name, str)

    def register_with_name(cls: Type[PullRequestMetricCalculator]):
        calculators[name] = cls
        return cls

    return register_with_name


class BinnedPullRequestMetricCalculator(Generic[T]):
    """Batched metrics calculation on sequential time intervals."""

    def __init__(self,
                 calcs: Sequence[PullRequestMetricCalculator[T]],
                 time_intervals: Sequence[datetime]):
        """
        Initialize a new instance of `BinnedPullRequestMetricCalculator`.

        :param calcs: Metric calculators. Their order matches the order of the results in \
                      `__call__()`.
        :param time_intervals: Time interval borders in UTC. Each interval spans \
                               `[time_intervals[i], time_intervals[i + 1]]`, the ending \
                               not included.
        """
        self.calcs = calcs
        assert len(time_intervals) >= 2
        self.time_intervals = time_intervals

    def __call__(self, items: Iterable[PullRequestTimes]) -> List[List[Metric[T]]]:
        """
        Calculate the binned metrics.

        For each time interval we collect the list of PRs that are relevant and measure \
        the specified metrics.
        """
        borders = self.time_intervals
        calcs = self.calcs
        dummy_bins = [[] for _ in borders[:-1]]
        items = sorted(items)
        regulars = [(calc, i) for i, calc in enumerate(calcs) if not calc.requires_full_span]
        full_spans = [(calc, i) for i, calc in enumerate(calcs) if calc.requires_full_span]
        regular_bins = self._bin_regulars(items) if regulars else dummy_bins
        full_span_bins = self._bin_full_spans(items) if full_spans else dummy_bins
        result = []
        for regular_bin, full_span_bin, time_from, time_to in zip(
                regular_bins, full_span_bins, borders, borders[1:]):
            for item in regular_bin:
                for calc, _ in regulars:
                    calc(item, time_from, time_to)
            for item in full_span_bin:
                for calc, _ in full_spans:
                    calc(item, time_from, time_to)
            values = [None] * len(calcs)
            for calc, i in chain(regulars, full_spans):
                values[i] = calc.value()
            result.append(values)
            for calc in calcs:
                calc.reset()
        return result

    def _bin_regulars(self, items: Iterable[PullRequestTimes]) -> List[List[PullRequestTimes]]:
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

    def _bin_full_spans(self, items: Iterable[PullRequestTimes]) -> List[List[PullRequestTimes]]:
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
