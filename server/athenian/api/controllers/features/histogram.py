from dataclasses import dataclass
from datetime import timedelta
from enum import IntEnum
from typing import Generic, List, TypeVar

import numpy as np

T = TypeVar("T", float, int, timedelta)


class Scale(IntEnum):
    """X axis scale: linear or logarithmic."""

    LINEAR = 1
    LOG = 2


@dataclass(frozen=True)
class Histogram(Generic[T]):
    """Definition of a histogram: X axis is `ticks` and Y axis is `bars`."""

    scale: Scale
    bins: int
    ticks: List[T]
    frequencies: List[int]


def calculate_histogram(samples: np.ndarray, scale: Scale, bins: int) -> Histogram[T]:
    """
    Calculate the histogram over the series of values.

    :param samples: Series of values which must be binned.
    :param scale: If LOG, we bin `log(samples)` and then exponent it back. If there are <= values,\
                  raise ValueError.
    :param bins: Number of bins. If 0, find the best number of bins.
    """
    assert isinstance(samples, np.ndarray)
    if len(samples) == 0:
        return Histogram(scale=scale, bins=0, ticks=[], frequencies=[])
    assert samples.dtype != np.dtype(object)
    try:
        unit, _ = np.datetime_data(samples.dtype)
        assert unit == "s"
    except TypeError:
        is_timedelta = False
    else:
        is_timedelta = True
        # saturate <1min >30days
        min = timedelta(minutes=1)
        month = timedelta(days=30)
        samples[samples < min] = min
        samples[samples > month] = month
        samples = samples.astype(int)
    if scale == Scale.LOG:
        if (samples <= 0).any():
            raise ValueError("Logarithmic scale is incompatible with non-positive samples: %s" %
                             samples[samples <= 0])
        samples = np.log(samples)
    if bins == 0:
        # find the best number of bins
        # "auto" uses Sturges which is worse than Doane
        fd_edges = np.histogram_bin_edges(samples, bins="fd")
        doane_edges = np.histogram_bin_edges(samples, bins="doane")
        bins = doane_edges if len(doane_edges) > len(fd_edges) else fd_edges
    hist, edges = np.histogram(samples, bins)
    if scale == Scale.LOG:
        edges = np.exp(edges)
    if is_timedelta:
        edges = edges.astype("timedelta64[s]")
        if edges[0] == timedelta(seconds=59):
            edges[0] = timedelta(seconds=60)
    return Histogram(scale=scale, bins=len(edges) - 1, ticks=edges.tolist(),
                     frequencies=hist.tolist())
