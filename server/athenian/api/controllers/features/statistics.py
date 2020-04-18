from datetime import timedelta
from functools import lru_cache, wraps
from typing import Sequence, Tuple

import bootstrapped.bootstrap as bootstrap
import bootstrapped.stats_functions as bs_stats
import numpy as np
import pandas as pd
import scipy.stats

from athenian.api.controllers.features.metric import T


vanilla_random_choice = np.random.choice
lru_random_choice = lru_cache(maxsize=1024)(np.random.choice)


@wraps(np.random.choice)
def cached_choice(*args, **kwargs):
    """Avoid hitting "unhashable type" error."""
    try:
        return lru_random_choice(*args, **kwargs)
    except TypeError:
        return vanilla_random_choice(*args, **kwargs)


np.random.choice = cached_choice


def mean_confidence_interval(data: Sequence[T], may_have_negative_values: bool, confidence=0.8,
                             ) -> Tuple[T, T, T]:
    """Calculate the mean value and the confidence interval."""
    assert len(data) > 0
    ns = 1_000_000_000
    max_conf_ratio = 100
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
                    confidence, len(arr) - 1, loc=m, scale=sem)
        else:
            # There used to be an assumption of the log-normal distribution here.
            # However, it worked terribly bad in practice. Besides, we never have values bigger
            # than a certain threshold (several years for timedelta-s).
            mci = bootstrap.bootstrap(arr, stat_func=bs_stats.mean, alpha=0.2)
            m, conf_min, conf_max = map(exact_type, (mci.value, mci.lower_bound, mci.upper_bound))
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


def median_confidence_interval(data: Sequence[T], confidence=0.8) -> Tuple[T, T, T]:
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
