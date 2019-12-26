from datetime import datetime
from typing import Dict, Generic, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
import scipy.stats

from athenian.api.controllers.features.metric import Metric, T
from athenian.api.controllers.miners.github.pull_request import PullRequestTimes


def mean_confidence_interval(data, confidence=0.95):
    """Calculate the mean value and the confidence interval."""
    arr = np.asarray(data)
    m, se = np.mean(arr), scipy.stats.sem(arr)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., len(arr) - 1)
    return m, m - h, m + h


def median_confidence_interval(data, confidence=0.95):
    """Calculate the median value and the confidence interval."""
    arr = np.asarray(data)
    arr = np.sort(arr)
    low_count, up_count = scipy.stats.binom.interval(confidence, arr.shape[0], 0.5, loc=0)
    low_count, up_count = int(low_count), int(up_count)
    # given this: https://onlinecourses.science.psu.edu/stat414/node/316
    # low_count and up_count both refer to W's value, W follows the binomial distribution.
    # low_count needs to be decremented, up_count no need to change in python indexing
    low_count -= 1
    return np.median(arr), arr[low_count], arr[up_count]


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

    def __call__(self, times: PullRequestTimes):
        """Supply another pull request timestamps to update the state."""
        sample = self.analyze(times)
        if sample is not None:
            self.samples.append(sample)

    def reset(self):
        """Reset the internal state."""
        self.samples.clear()

    def value(self) -> Metric[T]:
        """Calculate the current metric value."""
        raise NotImplementedError

    def analyze(self, times: PullRequestTimes) -> Optional[T]:
        """Do the actual state update."""
        raise NotImplementedError


class PullRequestAverageMetricCalculator(PullRequestMetricCalculator[T]):
    """Mean calculator."""

    def value(self) -> Metric[T]:
        """Calculate the current metric value."""
        if not self.samples:
            return Metric(False, 0, 0, 0)
        return Metric(True, *mean_confidence_interval(self.samples))

    def analyze(self, times: PullRequestTimes) -> Optional[T]:
        """Do the actual state update."""
        raise NotImplementedError


class PullRequestMedianMetricCalculator(PullRequestMetricCalculator[T]):
    """Median calculator."""

    def value(self) -> Metric[T]:
        """Calculate the current metric value."""
        if not self.samples:
            return Metric(False, 0, 0, 0)
        return Metric(True, *median_confidence_interval(self.samples))

    def analyze(self, times: PullRequestTimes) -> Optional[T]:
        """Do the actual state update."""
        raise NotImplementedError


calculators: Dict[str, Type[PullRequestMetricCalculator]] = {}


def register(name: str):
    """Keep track of the PR metric calculators."""
    def register_with_name(cls: Type[PullRequestMetricCalculator]):
        calculators[name] = cls
        return cls

    return register_with_name


class BinnedPullRequestMetricCalculator(Generic[T]):
    """Batched metrics calculation on sequential time intervals."""

    def __init__(self, calcs: Sequence[PullRequestMetricCalculator[T]],
                 time_intervals: Sequence[datetime]):
        """
        Initialize a new instance of `BinnedPullRequestMetricCalculator`.

        :param calcs: Metric calculators. Their order matches the order of the results in \
                      `__call__()`.
        :param time_intervals: Time interval borders. Each interval spans \
                               `[time_intervals[i], time_intervals[i + 1]]` (the ending is not \
                               included except for the last interval).
        """
        self.calcs = calcs
        self.time_intervals = time_intervals

    def __call__(self, items: Iterable[PullRequestTimes]) -> List[Tuple[Metric[T]]]:
        """
        Calculate the binned metrics.

        For each time interval we collect the list of PRs created then and measure the specified \
        metrics.
        """
        items = sorted(items, key=lambda x: x.created)
        if not items:
            return []
        borders = self.time_intervals
        if items[-1].created.best > borders[-1]:
            raise ValueError("there are PRs created after time_to")
        bins = [[] for _ in items]
        pos = 0
        for item in items:
            while item.created.best > borders[pos + 1]:
                pos += 1
            bins[pos].append(item)
        result = []
        calcs = self.calcs
        for bin in bins:
            for item in bin:
                for calc in calcs:
                    calc(item)
            result.append(tuple(calc.value() for calc in calcs))
            for calc in calcs:
                calc.reset()
        return result
