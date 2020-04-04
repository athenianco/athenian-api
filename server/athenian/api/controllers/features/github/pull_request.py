from datetime import date, datetime, timedelta, timezone
from typing import Dict, Generic, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
import scipy.stats

from athenian.api.controllers.features.metric import Metric, T
from athenian.api.controllers.miners.github.pull_request import PullRequestTimes


def mean_confidence_interval(data: Sequence[T], may_have_negative_values: bool, confidence=0.95,
                             ) -> Tuple[T, T, T]:
    """Calculate the mean value and the confidence interval."""
    assert len(data) > 0
    ns = 1_000_000_000
    max_conf_max_ratio = 10
    dtype_is_timedelta = isinstance(data[0], (pd.Timedelta, timedelta))
    if dtype_is_timedelta:
        # we have to convert the dtype because some ops required by scipy are missing
        # thus the precision is 1 second; otherwise there are integer overflows
        if isinstance(data[0], pd.Timedelta):
            arr = np.asarray([d.to_timedelta64() // ns for d in data], dtype=np.int64)
        else:
            arr = np.asarray(data, dtype="timedelta64[ns]").astype(np.int64) // ns
    else:
        arr = np.asarray(data)
    if may_have_negative_values:
        # assume a normal distribution
        m = np.mean(arr)
        if len(arr) == 1:
            # only one sample, we don't know the stddev so whatever value to indicate a failure
            conf_min = type(m)(0)
            conf_max = max_conf_max_ratio * m
        else:
            sem = scipy.stats.sem(arr)
            if sem == 0:
                conf_min = conf_max = m
            else:
                conf_min, conf_max = scipy.stats.t.interval(
                    confidence, len(arr) - 1, loc=m, scale=sem)
    else:
        # assume a log-normal distribution
        assert (arr >= 0).all()
        # deal with zeros - they are not allowed by log-normal
        zeros = arr == 0
        arr[zeros] = 1
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
            if logvar == 0:
                conf_min = conf_max = m
            else:
                logm = np.mean(logarr)
                d = np.sqrt(logvar / len(arr) + logvar**2 / (2 * (len(arr) - 1)))
                conf_min, conf_max = scipy.stats.t.interval(
                    confidence, len(arr) - 1, loc=logm + logvar / 2, scale=d)
                if conf_min > -20:
                    conf_min = np.exp(conf_min)
                else:
                    conf_min = 0
                if conf_max < logarr.max() + np.log(max_conf_max_ratio):
                    conf_max = np.exp(conf_max)
                else:
                    conf_max = arr.max() * max_conf_max_ratio
        if zeros.all():
            m -= 1
            conf_min = max(0, conf_min - 1)
            conf_max = max(m, conf_max - 1)
    if dtype_is_timedelta:
        # convert the dtype back
        m = pd.Timedelta(np.timedelta64(int(m * ns)))
        conf_min = pd.Timedelta(np.timedelta64(int(conf_min * ns)))
        conf_max = pd.Timedelta(np.timedelta64(int(conf_max * ns)))
        if not isinstance(data[0], pd.Timedelta):
            m = m.to_pytimedelta()
            conf_min = conf_min.to_pytimedelta()
            conf_max = conf_max.to_pytimedelta()
    else:
        # ensure the original dtype (can have been switched to float under the hood)
        dt = type(data[0])
        m = dt(m)
        conf_min = dt(conf_min)
        conf_max = dt(conf_max)
    return m, conf_min, conf_max


def median_confidence_interval(data: Sequence[T], confidence=0.95) -> Tuple[T, T, T]:
    """Calculate the median value and the confidence interval."""
    assert len(data) > 0
    arr = np.asarray(data)
    # The following code is based on:
    # https://onlinecourses.science.psu.edu/stat414/node/316
    arr = np.sort(arr)
    low_count, up_count = scipy.stats.binom.interval(confidence, arr.shape[0], 0.5)
    low_count, up_count = int(low_count), int(up_count)
    dt = type(data[0]) if type(data[0]) is not timedelta else lambda x: x
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
