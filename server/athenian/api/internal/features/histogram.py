from dataclasses import dataclass
from datetime import timedelta
from enum import IntEnum
from typing import Any, Generic, List, Optional, Tuple, TypeVar

import numpy as np

T = TypeVar("T", float, int, timedelta)


class Scale(IntEnum):
    """X axis scale: linear or logarithmic."""

    LINEAR = 1
    LOG = 2


@dataclass(slots=True, frozen=True)
class Histogram(Generic[T]):
    """Definition of a histogram: X axis is `ticks` and Y axis is `bars`."""

    scale: Scale
    bins: int
    ticks: List[T]
    frequencies: List[Any]
    interquartile: Tuple[T, T]


@dataclass(slots=True, frozen=True)
class HistogramParameters(Generic[T]):
    """Histogram attributes that define its shape."""

    scale: Optional[Scale]
    bins: Optional[int]
    ticks: Optional[Tuple[T]]


def calculate_histogram(
    samples: np.ndarray,
    scale: Optional[Scale],
    bins: Optional[int],
    ticks: Optional[list],
) -> Histogram[T]:
    """
    Calculate the histogram over the series of values.

    :param samples: Series of values which must be binned.
    :param scale: If LOG, we bin `log(samples)` and then exponent it back. If there are <= values,\
                  raise ValueError.
    :param bins: Number of bins. If 0, find the best number of bins.
    :param ticks: Fixed bin borders.
    """
    assert isinstance(samples, np.ndarray)
    if scale is None:
        scale = Scale.LINEAR
    if len(samples) == 0:
        if ticks is not None:
            return Histogram(
                scale=scale,
                bins=bins,
                ticks=ticks,
                frequencies=[0] * (len(ticks) - 1),
                interquartile=(0, 0),
            )
        return Histogram(scale=scale, bins=bins, ticks=[], frequencies=[], interquartile=(0, 0))
    assert samples.dtype != np.dtype(object)
    try:
        unit, _ = np.datetime_data(samples.dtype)
        assert unit == "s"
    except TypeError:
        is_timedelta = False
    else:
        is_timedelta = True
        # saturate <1min >30days
        min_ = timedelta(minutes=1)
        month = timedelta(days=30)
        samples[samples < min_] = min_
        samples[samples > month] = month
        samples = samples.astype(int)
        if ticks is not None:
            ticks = np.asarray(ticks, dtype="timedelta64[s]")
            ticks[ticks < min_] = min_
            ticks[ticks > month] = month
            ticks = ticks.astype(int)
    iq = np.quantile(samples, [0.25, 0.75])
    if is_timedelta:
        iq = iq.astype("timedelta64[s]")
    if scale == Scale.LOG:
        if (samples <= 0).any():
            raise ValueError(
                "Logarithmic scale is incompatible with non-positive samples: %s"
                % samples[samples <= 0],
            )
        samples = np.log(samples)
    if ticks is not None:
        min_bin, max_bin = samples.min(), samples.max()
        bins = []
        if min_bin < ticks[0]:
            bins.append([min_bin])
        bins.append(ticks)
        if max_bin > ticks[-1]:
            bins.append([max_bin])

        bins = np.concatenate(bins)
    elif not bins:
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
    return Histogram(
        scale=scale,
        bins=len(edges) - 1,
        ticks=edges.tolist(),
        frequencies=hist.tolist(),
        interquartile=tuple(iq),
    )
