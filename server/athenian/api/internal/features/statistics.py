from datetime import timedelta
from functools import wraps
from typing import Tuple

import bootstrapped.bootstrap as bootstrap
import bootstrapped.stats_functions as bs_stats
import numpy as np
import scipy.stats

from athenian.api.internal.features.metric import T


class NumpyRandomChoiceCache:
    """
    Cache numpy.random.choice when `a` is an integer and `size` is a 2D shape.

    This caching improves the performance of bootstrapped because we don't have to generate a big
    random array every time.
    """

    vanilla_random_choice = np.random.choice

    def __init__(self, n: int):
        """
        Generate n random integers on startup.

        They will be re-used every time we random.choice with replacement.
        """
        self._generate(n)

    def _generate(self, n: int):
        np.random.seed(777)
        self.entropy = np.random.randint(low=np.iinfo(np.uint32).max + 1, size=n, dtype=np.uint32)

    @wraps(np.random.choice)
    def __call__(self, a, size=None, replace=True, p=None):
        """Pretend to be np.random.choice."""
        if isinstance(a, (int, np.int32, np.int64)) and replace and p is None:
            full_size = np.prod(size)
            if full_size > len(self.entropy):
                self._generate(full_size)
            return self.entropy[:full_size].reshape(size) % a
        return self.vanilla_random_choice(a, size=size, replace=replace, p=p)


np.random.choice = NumpyRandomChoiceCache(2_000_000)  # +8MB


def mean_confidence_interval(
    data: np.ndarray,
    may_have_negative_values: bool,
    confidence=0.8,
) -> Tuple[T, T, T]:
    """Calculate the mean value and the confidence interval."""
    assert len(data) > 0
    assert isinstance(data, np.ndarray)
    max_conf_ratio = 100
    assert data.dtype != np.dtype(object)
    try:
        unit, _ = np.datetime_data(data.dtype)
        assert unit == "s"
    except TypeError:
        dtype_is_timedelta = False
        arr = data
    else:
        dtype_is_timedelta = True
        # we have to convert the dtype because some ops required by scipy are missing
        # thus the precision is 1 second; otherwise there are integer overflows
        arr = data.astype(np.int64)
    exact_type = type(arr[0])
    if len(arr) == 1:
        # we don't know the stddev so whatever value to indicate poor confidence
        m = arr[0]
        conf_min = exact_type(m / max_conf_ratio)
        conf_max = max_conf_ratio * m
    else:
        if may_have_negative_values:
            # assume a normal distribution
            m = np.mean(arr)
            sem = scipy.stats.sem(arr, ddof=1)
            if sem == 0:
                conf_min = conf_max = m
            else:
                conf_min, conf_max = scipy.stats.t.interval(
                    confidence, len(arr) - 1, loc=m, scale=sem
                )
        else:
            # There used to be an assumption of the log-normal distribution here.
            # However, it worked terribly bad in practice. Besides, we never have values bigger
            # than a certain threshold (several years for timedelta-s).

            # Reduce the number of iterations to keep the constant amount of work.
            num_iterations = max(int(2000 * min(1.0, 200 / len(arr))), 10)
            for _ in range(10):
                mci = bootstrap.bootstrap(
                    arr, num_iterations=num_iterations, stat_func=bs_stats.mean, alpha=0.2
                )
                m, conf_min, conf_max = map(
                    exact_type, (mci.value, mci.lower_bound, mci.upper_bound)
                )
                if conf_min >= 0:
                    break
                num_iterations *= 2
            if conf_min < 0:
                conf_min = 0
    if dtype_is_timedelta:
        # convert the dtype back
        m = timedelta(seconds=int(m))
        conf_min = timedelta(seconds=int(conf_min))
        conf_max = timedelta(seconds=int(conf_max))
    else:

        def type_conv(x):
            return data.dtype.type(x).item()

        m = type_conv(m)
        conf_min = type_conv(conf_min)
        conf_max = type_conv(conf_max)
    return m, conf_min, conf_max


def median_confidence_interval(data: np.ndarray, confidence=0.8) -> Tuple[T, T, T]:
    """Calculate the median value and the confidence interval."""
    assert len(data) > 0
    assert isinstance(data, np.ndarray)
    assert data.dtype != np.dtype(object)
    # The following code is based on:
    # https://onlinecourses.science.psu.edu/stat414/node/316
    arr = np.sort(data)
    low_count, up_count = scipy.stats.binom.interval(confidence, arr.shape[0], 0.5)
    low_count, up_count = int(low_count), int(up_count)

    def type_conv(x):
        return data.dtype.type(x).item()

    return type_conv(np.median(arr)), type_conv(arr[low_count]), type_conv(arr[up_count - 1])
