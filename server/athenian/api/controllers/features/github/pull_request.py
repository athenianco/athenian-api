from datetime import date, datetime, timezone
from typing import Dict, Generic, Iterable, List, Optional, Sequence, Tuple, Type

from athenian.api.controllers.features.metric import Metric, T
from athenian.api.controllers.features.statistics import mean_confidence_interval, \
    median_confidence_interval
from athenian.api.controllers.miners.github.pull_request import PullRequestTimes


class PullRequestMetricCalculator(Generic[T]):
    """
    Pull request metric calculator, base abstract class.

    Each call to `PullRequestMetricCalculator()` feeds another PR to update the state.
    `PullRequestMetricCalculator.value()` returns the current metric value.
    `PullRequestMetricCalculator.analyze()` is to be implemented by the particular metric
    calculators.
    """

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
        if not self.may_have_negative_values and not any(self.samples):
            # The log-normal distribution is not compatible with this bullshit.
            return Metric(False, None, None, None)
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

    def __init__(self, calcs: Sequence[PullRequestMetricCalculator[T]],
                 time_intervals: Sequence[date]):
        """
        Initialize a new instance of `BinnedPullRequestMetricCalculator`.

        :param calcs: Metric calculators. Their order matches the order of the results in \
                      `__call__()`.
        :param time_intervals: Time interval borders. Each interval spans \
                               `[time_intervals[i], time_intervals[i + 1]]`, and the ending is \
                               not included except for the last interval.
        """
        self.calcs = calcs
        assert len(time_intervals) >= 2
        self.time_intervals = [datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
                               for d in time_intervals[:-1]]
        self.time_intervals.append(
            datetime.combine(time_intervals[-1], datetime.max.time(), tzinfo=timezone.utc))

    def __call__(self, items: Iterable[PullRequestTimes]) -> List[Tuple[Metric[T]]]:
        """
        Calculate the binned metrics.

        For each time interval we collect the list of PRs that are relevant and measure \
        the specified metrics.
        """
        borders = self.time_intervals
        calcs = self.calcs
        bins = [[] for _ in borders[:-1]]
        items = sorted(items, key=lambda x: x.created.best)
        pos = 0
        for item in items:
            while pos < len(borders) - 1 and item.created.best > borders[pos + 1]:
                pos += 1
            endpoint = item.max_timestamp()
            span = pos
            while span < len(bins) and endpoint > borders[span]:
                bins[span].append(item)
                span += 1
        result = []
        for bin, time_from, time_to in zip(bins, borders, borders[1:]):
            for item in bin:
                for calc in calcs:
                    calc(item, time_from, time_to)
            result.append(tuple(calc.value() for calc in calcs))
            for calc in calcs:
                calc.reset()
        return result
