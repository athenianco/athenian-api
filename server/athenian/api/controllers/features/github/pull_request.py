from datetime import datetime
from typing import Dict, Generic, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
import scipy.stats

from athenian.api.controllers.features.metric import Metric, T
from athenian.api.controllers.miners.github.pull_request import PullRequestTimes


def mean_confidence_interval(
        data: Sequence[T], may_have_negative_values: bool, confidence=0.95,
        ) -> Tuple[Optional[T], Optional[T], Optional[T]]:
    """Calculate the mean value and the confidence interval."""
    if len(data) == 0:
        return None, None, None
    timedelta = isinstance(data[0], pd.Timedelta)
    ns = 1_000_000_000
    max_conf_max_ratio = 10
    if timedelta:
        # thus the precision is 1 second; otherwise there are integer overflows
        arr = np.asarray([d.to_timedelta64() // ns for d in data], dtype=np.int64)
    else:
        arr = np.asarray(data)
    if may_have_negative_values:
        # assume a normal distribution
        m = np.mean(arr)
        sem = scipy.stats.sem(arr)
        if sem != sem:
            # only one sample, we don't know the stddev so whatever value to indicate a failure
            conf_min = type(m)(0)
            conf_max = max_conf_max_ratio * m
        else:
            conf_min, conf_max = scipy.stats.t.interval(confidence, len(arr) - 1, loc=m, scale=sem)
    else:
        # assume a log-normal distribution
        assert (arr >= 0).all()
        arr = arr[arr != 0]
        m = np.mean(arr)
        if len(arr) == 1:
            # only one sample, we don't know the stddev so whatever value to indicate a failure
            conf_min = type(m)(0)
            conf_max = max_conf_max_ratio * m
        else:
            # modified Cox method
            # https://doi.org/10.1080/10691898.2005.11910638
            logarr = np.log(arr)
            logvar = np.var(logarr, ddof=1)
            logm = np.mean(logarr)
            d = np.sqrt(logvar / len(arr) + logvar**2 / (2 * (len(arr) - 1)))
            conf_min, conf_max = scipy.stats.t.interval(
                confidence, len(arr) - 1, loc=logm + logvar / 2, scale=d)
            conf_min, conf_max = np.exp(conf_min), np.exp(conf_max)
            if conf_max / m > max_conf_max_ratio:
                conf_max = max_conf_max_ratio * m
    if timedelta:
        # convert back to pd.Timedelta-s
        m = pd.Timedelta(np.timedelta64(int(m * ns)))
        conf_min = pd.Timedelta(np.timedelta64(int(conf_min * ns)))
        conf_max = pd.Timedelta(np.timedelta64(int(conf_max * ns)))
    else:
        dt = type(data[0])
        m = dt(m)
        conf_min = dt(conf_min)
        conf_max = dt(conf_max)
    return m, conf_min, conf_max


def median_confidence_interval(data: Sequence[T], confidence=0.95,
                               ) -> Tuple[Optional[T], Optional[T], Optional[T]]:
    """Calculate the median value and the confidence interval."""
    if len(data) == 0:
        return None, None, None
    arr = np.asarray(data)
    # https://onlinecourses.science.psu.edu/stat414/node/316
    arr = np.sort(arr)
    low_count, up_count = scipy.stats.binom.interval(confidence, arr.shape[0], 0.5)
    low_count, up_count = int(low_count), int(up_count)
    dt = type(data[0])
    return dt(np.median(arr)), dt(arr[low_count]), dt(arr[up_count - 1])


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
    may_have_negative_values: bool

    def value(self) -> Metric[T]:
        """Calculate the current metric value."""
        if not self.samples:
            return Metric(False, 0, 0, 0)
        assert self.may_have_negative_values is not None
        return Metric(True, *mean_confidence_interval(self.samples, self.may_have_negative_values))

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
        items = sorted(items, key=lambda x: x.created.best)
        if not items:
            return []
        borders = self.time_intervals
        if items[-1].created.best > borders[-1]:
            raise ValueError("there are PRs created after time_to")
        bins = [[] for _ in borders]
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
