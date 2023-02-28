from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import dataclasses
from datetime import datetime, timedelta
from graphlib import TopologicalSorter
from itertools import chain
from typing import (
    Any,
    Callable,
    Collection,
    Generic,
    Iterable,
    KeysView,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
)
import warnings

import numpy as np
from numpy import typing as npt
import pandas as pd
import sentry_sdk

from athenian.api.int_to_str import int_to_str
from athenian.api.internal.features.histogram import Histogram, Scale, calculate_histogram
from athenian.api.internal.features.metric import (
    Metric,
    MetricFloat,
    MetricInt,
    MultiMetric,
    NumpyMetric,
    T,
    make_metric,
)
from athenian.api.internal.features.statistics import (
    mean_confidence_interval,
    median_confidence_interval,
)
from athenian.api.internal.logical_repos import coerce_logical_repos
from athenian.api.internal.miners.types import PR_JIRA_DETAILS_COLUMN_MAP
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings
from athenian.api.models.metadata.jira import Issue
from athenian.api.sparse_mask import SparseMask
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import dataclass_asdict
from athenian.api.unordered_unique import in1d_str, unordered_unique

DEFAULT_QUANTILE_STRIDE = 14


class MetricCalculator(Generic[T], ABC):
    """
    Arbitrary type T metric calculator, base abstract class.

    Each call to `MetricCalculator()` feeds another input object to update the state.
    `MetricCalculator.value()` returns the current metric value.
    `MetricCalculator._analyze()` is to be implemented by the particular metric calculators.
    """

    # Types of dependencies - upstream MetricCalculator-s.
    deps: tuple[Type["MetricCalculator"], ...] = ()

    # specific Metric class
    metric: Type[NumpyMetric] = None

    # _analyze() may return arrays of non-standard shape, `samples` are ignored
    is_pure_dependency = False

    @property
    def dtype(self) -> np.dtype:
        """Return the metric value's dtype."""
        return self.metric.dtype["value"]

    @property
    def nan(self) -> Any:
        """Return the metric's Not-a-Number-like value."""
        return self.metric.nan

    @property
    def has_nan(self) -> bool:
        """Return the value indicating whether the metric has a native Not-a-Number-like."""
        nan = self.metric.nan
        return nan != nan

    @property
    def calcs(self) -> list[MetricCalculator | list[MetricCalculator]]:
        """Return the dependencies."""
        return self._calcs

    def __init__(
        self,
        *deps: MetricCalculator | tuple[MetricCalculator] | list[MetricCalculator],
        quantiles: Sequence[float],
        **kwargs,
    ):
        """Initialize a new `MetricCalculator` instance."""
        self.reset()
        self._calcs: list[MetricCalculator | list[MetricCalculator]] = []
        self._quantiles = tuple(quantiles)
        assert len(self._quantiles) == 2
        assert self._quantiles[0] >= 0
        assert self._quantiles[1] <= 1
        assert self._quantiles[0] <= self._quantiles[1]
        self.__dict__.update(kwargs)
        for calc in deps:
            if isinstance(calc, (list, tuple)):
                example = calc[0]
                if len(calc) == 1:
                    calc = example
            else:
                example = calc
            for cls in self.deps:
                if isinstance(example, cls):
                    self._calcs.append(calc)
                    break
            else:
                raise AssertionError("%s is not listed in the dependencies" % type(calc))

    def __call__(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        quantiles_mounted_at: Optional[int],
        groups_mask: np.ndarray,
        **kwargs,
    ) -> None:
        """Calculate the samples over the supplied facts.

        :param facts: Mined facts about pull requests: timestamps, stats, etc.
        :param min_times: Beginnings of the considered time intervals. They are needed to discard \
                          samples with both ends less than the minimum time.
        :param max_times: Endings of the considered time intervals. They are needed to discard \
                          samples with both ends greater than the maximum time.
        :param quantiles_mounted_at: Time intervals that represent the population split start
            with this index. We calculate the quantiles separately for each such interval \
            and independently discard the outliers.
        :param groups_mask: len(groups) * len(facts) boolean array that chooses the elements \
                            of each group.
        """
        assert isinstance(facts, pd.DataFrame)
        assert isinstance(min_times, np.ndarray)
        assert isinstance(max_times, np.ndarray)
        assert min_times.shape == max_times.shape
        assert min_times.dtype == max_times.dtype
        assert len(min_times.shape) == 1
        try:
            assert np.datetime_data(min_times.dtype)[0] == "ns"
        except TypeError:
            raise AssertionError("min_times must be of datetime64[ns] dtype")
        if has_quantiles := self._quantiles != (0, 1):
            assert quantiles_mounted_at is not None
        if quantiles_mounted_at is not None:
            assert len(min_times) > quantiles_mounted_at

        assert isinstance(groups_mask, np.ndarray)
        assert len(groups_mask.shape) == 2
        assert len(groups_mask) > 0 or facts.empty
        if facts.empty:
            self._peek = np.empty((len(min_times), 0), object)
            ts_dim = quantiles_mounted_at or len(min_times)
            self._grouped_notnull = self._grouped_sample_mask = SparseMask.empty(
                len(groups_mask), ts_dim, 0,
            )
            if not self.is_pure_dependency:
                self._split_samples(
                    np.array([], self.dtype), SparseMask.empty(len(groups_mask), ts_dim, 0),
                )
            return
        assert groups_mask.shape[1] == len(facts)
        self._peek = peek = self._analyze(facts, min_times, max_times, **kwargs)
        assert isinstance(peek, np.ndarray), type(self)
        if self.is_pure_dependency:
            return
        assert peek.shape == (len(min_times), len(facts)), (peek.shape, type(self))
        groups_mask = SparseMask.from_dense(groups_mask)
        if has_quantiles:
            discard_mask = self._calculate_discard_mask(peek, quantiles_mounted_at, groups_mask)
        if quantiles_mounted_at is not None:
            peek = peek[:quantiles_mounted_at]
        notnull = SparseMask.from_dense(self._find_notnull(peek))[None, :].repeat(
            len(groups_mask), 0,
        )
        group_sample_mask = groups_mask[:, None, :].repeat(notnull.shape[1], 1)
        self._grouped_notnull = group_sample_mask = group_sample_mask & notnull
        del notnull
        if has_quantiles:
            group_sample_mask = group_sample_mask.copy()
            discard_mask = discard_mask[:, None, :].repeat(group_sample_mask.shape[1], axis=1)
            self._grouped_sample_mask = group_sample_mask = group_sample_mask - discard_mask
            del discard_mask
        else:
            self._grouped_sample_mask = self._grouped_notnull
        sample_group_levels = group_sample_mask.unravel()
        flat_samples = peek.astype(self.dtype, copy=False)[sample_group_levels[1:]]
        del sample_group_levels
        self._split_samples(flat_samples, group_sample_mask)

    def _split_samples(self, flat_samples: np.ndarray, group_sample_mask: SparseMask) -> None:
        group_sample_lengths = group_sample_mask.sum_last_axis()
        self._samples = np.full(np.prod(group_sample_mask.shape[:2]), None, object)
        self._samples[:] = np.split(flat_samples, np.cumsum(group_sample_lengths.ravel())[:-1])
        self._samples = self._samples.reshape(group_sample_mask.shape[:2])

    @property
    def values(self) -> list[list[Metric[T]]]:
        """
        Calculate the current metric values.

        :return: groups x times -> metric.
        """
        if self._last_values is None:
            self._last_values = self._values()
        return self._last_values

    @property
    def peek(self) -> np.ndarray:
        """
        Return the last calculated samples, with None-s: time intervals x facts.

        The shape is time intervals x facts - that is, before grouping and discarding outliers.
        """
        return self._peek

    @property
    def samples(self) -> np.ndarray:
        """Return the last calculated samples, without None-s: groups x time intervals x facts.

        The quantiles are already applied.
        """
        return self._samples

    @property
    def grouped_notnull(self) -> SparseMask:
        """Return the last calculated boolean mask of shape groups x time intervals x facts \
        that selects non-None samples *before discarding outliers by quantiles*.

        This mask is typically consumed by the counters that must ignore the quantiles.
        """
        return self._grouped_notnull

    @property
    def grouped_sample_mask(self) -> SparseMask:
        """Return the last calculated boolean mask of shape groups x time intervals x facts \
        that selects non-None samples *after discarding outliers by quantiles*."""
        return self._grouped_sample_mask

    def reset(self) -> None:
        """Clear the current state of the calculator."""
        self._samples = np.empty((0, 0), dtype=object)
        self._peek = np.empty((0, 0), dtype=object)
        self._grouped_notnull = self._grouped_sample_mask = SparseMask.empty(0, 0, 0)
        self._last_values = None

    def split(self) -> list["MetricCalculator"]:
        """Replicate yourself depending on the previously set external keyword arguments."""
        return [self]

    def clone(self, *args, **kwargs) -> "MetricCalculator":
        """Clone from an existing calculator."""
        instance = object.__new__(type(self))
        instance.__dict__ = self.__dict__.copy()
        instance.__init__(*args, **kwargs)
        return instance

    @abstractmethod
    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Calculate the samples for each item in the data frame.

        :return: np.ndarray[(len(times), len(facts))]
        """
        raise NotImplementedError

    @abstractmethod
    def _value(self, samples: np.ndarray) -> Metric[T]:
        """Calculate the metric values from the current samples."""
        raise NotImplementedError

    def _values(self) -> list[list[Metric[T]]]:
        return [[self._value(s) for s in gs] for gs in self._samples]

    @classmethod
    def _find_notnull(cls, peek: np.ndarray) -> npt.NDArray[bool]:
        if peek.dtype is np.dtype(object):
            # this is the slowest, avoid as much as possible
            return peek != np.array(None)
        nan = cls.metric.nan
        if nan != nan:
            return peek == peek
        if nan is not None:
            return peek != nan
        return peek != peek.dtype.type(0)

    def _calculate_discard_mask(
        self,
        peek: np.ndarray,
        quantiles_mounted_at: int,
        groups_mask: SparseMask,
    ) -> SparseMask:
        if peek.dtype != object:
            dtype = peek.dtype
        else:
            dtype = self.dtype
        qnotnull = self._find_notnull(peek[quantiles_mounted_at:])
        qnotnull = SparseMask.from_dense(qnotnull)[None, :].repeat(len(groups_mask), 0)
        group_sample_mask = qnotnull & groups_mask[:, None, :].repeat(qnotnull.shape[1], 1)
        flat_samples = np.broadcast_to(
            peek.astype(dtype, copy=False)[None, quantiles_mounted_at:], qnotnull.shape,
        )[group_sample_mask.unravel()].ravel()
        group_sample_lengths = group_sample_mask.sum_last_axis()
        if self.has_nan:
            max_len = group_sample_lengths.max()
            if max_len == 0:
                cut_values = np.full((*group_sample_lengths.shape, 2), None, dtype)
            else:
                surrogate_samples = np.full((*group_sample_lengths.shape, max_len), None, dtype)
                flat_lengths = group_sample_lengths.ravel().astype(int, copy=False)
                flat_mask = flat_lengths > 0
                flat_lengths = flat_lengths[flat_mask]
                flat_offsets = (np.arange(group_sample_lengths.size) * max_len)[
                    flat_mask
                ] + flat_lengths
                flat_cumsum = flat_lengths.cumsum()
                indexes = np.arange(flat_cumsum[-1]) + np.repeat(
                    flat_offsets - flat_cumsum, flat_lengths,
                )
                surrogate_samples.ravel()[indexes] = flat_samples
                with warnings.catch_warnings():
                    # this will happen if some groups are all NaN-s, that's normal
                    warnings.filterwarnings("ignore", "All-NaN slice encountered")
                    cut_values = np.moveaxis(
                        np.nanquantile(
                            surrogate_samples, self._quantiles, method="nearest", axis=-1,
                        ).astype(dtype),
                        0,
                        -1,
                    )
        else:
            pos = 0
            cut_values = np.zeros((group_sample_lengths.size, 2), dtype=dtype)
            for i, slen in enumerate(group_sample_lengths.ravel()):
                if slen > 0:
                    cut_values[i] = np.quantile(
                        flat_samples[pos : pos + slen], self._quantiles, method="nearest",
                    )
                    pos += slen
        cut_values = cut_values.reshape((*group_sample_lengths.shape, 2))
        grouped_qpeek = np.broadcast_to(peek[None, quantiles_mounted_at:], qnotnull.shape)
        group_qmask = groups_mask[:, None, :].repeat(grouped_qpeek.shape[1], 1)
        group_qmask_unravel = group_qmask.unravel()
        grouped_qpeek_nnz = grouped_qpeek[group_qmask_unravel]
        # fmt: off
        if self._quantiles[0] > 0:
            left_mask = grouped_qpeek_nnz < cut_values[:, :, 0][group_qmask_unravel[:-1]]
            discard_left_mask = SparseMask(
                group_qmask.indexes[left_mask], group_qmask.shape,
            ).any(axis=1)
        if self._quantiles[1] < 1:
            right_mask = grouped_qpeek_nnz > cut_values[:, :, 1][group_qmask_unravel[:-1]]
            discard_right_mask = SparseMask(
                group_qmask.indexes[right_mask], group_qmask.shape,
            ).any(axis=1)
        # fmt: on
        if self._quantiles[0] == 0:
            discard_mask = discard_right_mask
        elif self._quantiles[1] == 1:
            discard_mask = discard_left_mask
        else:
            discard_mask = discard_left_mask | discard_right_mask
        return discard_mask


class WithoutQuantilesMixin:
    """Ignore the quantiles."""

    def __init__(self, *deps: "MetricCalculator", quantiles: Sequence[float], **kwargs):
        """Override the constructor to disable the quantiles."""
        super().__init__(*deps, quantiles=(0, 1), **kwargs)


class AverageMetricCalculator(MetricCalculator[T], ABC):
    """Mean calculator."""

    may_have_negative_values: bool

    def _value(self, samples: np.ndarray) -> Metric[T]:
        if len(samples) == 0:
            return self.metric.from_fields(False, None, None, None)
        assert self.may_have_negative_values is not None
        if not self.may_have_negative_values:
            zero = samples.dtype.type(0)
            negative = np.nonzero(samples < zero)[0]
            try:
                assert len(negative) == 0, (samples[negative], negative, type(self).__name__)
            except AssertionError as e:
                if sentry_sdk.Hub.current.scope.transaction is not None:
                    sentry_sdk.capture_exception(e)
                    samples = samples[samples >= zero]
                else:
                    raise e from None
        return self.metric.from_fields(
            True, *mean_confidence_interval(samples, self.may_have_negative_values),
        )


class MedianMetricCalculator(WithoutQuantilesMixin, MetricCalculator[T], ABC):
    """Median calculator."""

    def _value(self, samples: np.ndarray) -> Metric[T]:
        if len(samples) == 0:
            return self.metric.from_fields(False, None, None, None)
        return self.metric.from_fields(True, *median_confidence_interval(samples))


class AggregationMetricCalculator(WithoutQuantilesMixin, MetricCalculator[T], ABC):
    """Any simple array aggregation calculator."""

    # value indicating whether the behavior of _agg() matches np.*.reduceat()
    # particularly, carry the previous value if the offsets are equal
    is_reduceat = False

    def _value(self, samples: np.ndarray) -> Metric[T]:
        return self.metric.from_fields(True, samples, None, None)

    def _split_samples(self, flat_samples: np.ndarray, group_sample_mask: SparseMask) -> None:
        group_sample_lengths = np.trim_zeros(group_sample_mask.sum_last_axis().ravel(), "b")
        self._samples = np.zeros(group_sample_mask.shape[:2], dtype=self.dtype)
        if len(group_sample_lengths) == 0:
            return
        samples = self._samples.ravel()
        group_sample_offsets = np.zeros(group_sample_lengths.size + 1, dtype=int)
        np.cumsum(group_sample_lengths, out=group_sample_offsets[1:])
        samples[: len(group_sample_lengths)] = self._agg(flat_samples, group_sample_offsets[:-1])
        if self.is_reduceat:
            samples[np.flatnonzero(group_sample_lengths == 0)] = np.dtype(self.dtype).type()

    @abstractmethod
    def _agg(self, samples: np.ndarray, offsets: np.ndarray) -> T:
        raise NotImplementedError()


class SumMetricCalculator(AggregationMetricCalculator[T], ABC):
    """Sum calculator."""

    is_reduceat = True
    _agg = np.add.reduceat


class MaxMetricCalculator(AggregationMetricCalculator[T], ABC):
    """Maximum calculator."""

    is_reduceat = True
    _agg = np.maximum.reduceat


class AnyMetricCalculator(AggregationMetricCalculator[T], ABC):
    """len(samples) > 0 calculator."""

    def _agg(self, samples: np.ndarray, offsets: np.ndarray) -> T:
        return np.diff(offsets.base) > 0


class Counter(MetricCalculator[int], ABC):
    """Count the number of items that were used to calculate the specified metric."""

    metric = MetricInt

    def __call__(self, *args, **kwargs) -> None:
        """Copy by reference the same peek and samples from the only dependency."""
        calc = self._calcs[0]
        self._peek = calc.peek
        self._grouped_notnull = calc.grouped_notnull
        self._grouped_sample_mask = calc.grouped_sample_mask
        self._samples = calc.samples

    def _values(self) -> list[list[Metric[T]]]:
        if self._quantiles != (0, 1):
            # if we've got the quantiles, report the lengths
            return [
                [self.metric.from_fields(True, len(s), None, None) for s in gs]
                for gs in self.samples
            ]
        # otherwise, ignore the upstream quantiles
        return [
            [self.metric.from_fields(True, s, None, None) for s in gs]
            for gs in self.grouped_notnull.dense().sum(axis=-1)
        ]

    def _value(self, samples: np.ndarray) -> Metric[int]:
        raise AssertionError("this must be never called")

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        raise AssertionError("this must be never called")


class HistogramCalculator(MetricCalculator, ABC):
    """Pull request histogram calculator, base abstract class."""

    def histogram(
        self,
        scale: Optional[Scale],
        bins: Optional[int],
        ticks: Optional[list],
    ) -> list[list[Histogram[T]]]:
        """Calculate the histogram over the current distribution."""
        histograms = []
        for group_samples in self.samples:
            histograms.append(group_histograms := [])
            for samples in group_samples:
                if scale == Scale.LOG:
                    if (shift_log := getattr(self, "_shift_log", None)) is not None:
                        samples = shift_log(samples)
                group_histograms.append(calculate_histogram(samples, scale, bins, ticks))
        return histograms


class MetricCalculatorEnsemble:
    """Several calculators ordered in sequence such that the dependencies are respected and each \
    calculator class is calculated only once."""

    def __init__(
        self,
        *metrics: str,
        class_mapping: dict[str, Type[MetricCalculator]],
        quantiles: Sequence[float],
        quantile_stride: int,
        **kwargs,
    ):
        """Initialize a new instance of MetricCalculatorEnsemble class."""
        metric_classes = defaultdict(list)
        for m in metrics:
            metric_classes[class_mapping[m]].append(m)
        self._calcs, self._metrics = self._plan_classes(metric_classes, quantiles, **kwargs)
        self._quantiles = tuple(quantiles)
        self._quantile_stride = quantile_stride
        if self._quantiles != (0, 1):
            assert self._quantile_stride

    def __getitem__(self, metric: str) -> list[MetricCalculator]:
        """Return the owned calculator for the given metric."""
        return self._metrics[metric]

    @staticmethod
    def compose_groups_mask(groups: Sequence[Sequence[int]], len_facts: int) -> np.ndarray:
        """Convert group indexes to group masks."""
        groups_mask = np.zeros((len(groups), len_facts), dtype=bool)
        for i, g in enumerate(groups):
            groups_mask[i, g] = True
        return groups_mask

    @staticmethod
    def _plan_classes(
        metric_classes: dict[Type[MetricCalculator], list[str]],
        quantiles: Sequence[float],
        **kwargs,
    ) -> tuple[list[MetricCalculator], dict[str, list[MetricCalculator]]]:
        dig = {}
        required_classes = list(metric_classes)
        while required_classes:
            cls = required_classes.pop()
            if cls.deps:
                for dep in cls.deps:
                    if dep not in dig:
                        required_classes.append(dep)
                    dig.setdefault(cls, []).append(dep)
            else:
                dig.setdefault(cls, [])
        calcs = []
        metrics = {}
        cls_instances = {}
        for cls in TopologicalSorter(dig).static_order():
            calc = cls(*(cls_instances[dep] for dep in cls.deps), quantiles=quantiles, **kwargs)
            calcs.extend(clones := calc.split())
            cls_instances[cls] = clones
            try:
                for metric in metric_classes[cls]:
                    metrics[metric] = clones
            except KeyError:
                continue
        return calcs, metrics

    @staticmethod
    def compose_quantile_time_intervals(
        min_time: np.datetime64,
        max_time: np.datetime64,
        quantile_stride: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the additional time intervals needed to filter out the local outliers."""
        zero = np.datetime64("2000-01-03")
        min_qli = (min_time.astype("datetime64[D]") - zero) // np.timedelta64(quantile_stride, "D")
        max_qli = (
            ((max_time + np.timedelta64(23, "h")).astype("datetime64[D]") - zero)
            + np.timedelta64(quantile_stride - 1, "D")
        ) // np.timedelta64(quantile_stride, "D")
        timeline = (
            zero
            + (min_time - min_time.astype("datetime64[D]"))
            + (np.arange(min_qli, max_qli + 1, 1) * quantile_stride).astype("timedelta64[D]")
        )
        return timeline[:-1], timeline[1:]

    def _compose_quantile_time_intervals(
        self,
        min_time: np.datetime64,
        max_time: np.datetime64,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.compose_quantile_time_intervals(min_time, max_time, self._quantile_stride)

    @sentry_span
    def __call__(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        groups: Sequence[Sequence[int]],
        **kwargs,
    ) -> None:
        """Invoke all the owned metric calculators on the same input."""
        groups_mask = self.compose_groups_mask(groups, len(facts))
        if self._quantiles == (0, 1):
            quantiles_mounted_at = None
        else:
            quantiles_mounted_at = len(min_times)
            qmins, qmaxs = self._compose_quantile_time_intervals(min_times.min(), max_times.max())
            min_times = np.concatenate([min_times, qmins])
            max_times = np.concatenate([max_times, qmaxs])
        for calc in self._calcs:
            calc(facts, min_times, max_times, quantiles_mounted_at, groups_mask, **kwargs)

    def __bool__(self) -> bool:
        """Return True if there is at leat one calculator inside; otherwise, False."""
        return bool(len(self._calcs))

    @sentry_span
    def values(self) -> dict[str, list[list[Metric[T]]]]:
        """
        Calculate the current metric values.

        :return: Mapping from metric name to corresponding metric values (groups x times).
        """
        result = {}
        for metric, calcs in self._metrics.items():
            values = [calc.values for calc in calcs]
            if len(values) == 1:
                values = values[0]
            else:
                values = [[MultiMetric(*vt) for vt in zip(*vg)] for vg in zip(*values)]
            result[metric] = values
        return result

    def reset(self) -> None:
        """Clear the states of all the contained calculators."""
        [c.reset() for c in self._calcs]


MetricComparisonRatio = make_metric("MetricComparisonRatio", __name__, np.float32, -1)


class ThresholdComparisonRatioCalculator(AggregationMetricCalculator[float]):
    """Calculate the ratio of metric values satisfying a given threshold.

    Metric values are computed by the upstream MetricCalculator dependency.

    Subclasses must specify the numpy comparison function
    (eg. np.less_equal, np.greater, ...) to use between the metric
    values and the threshold.

    """

    metric = MetricComparisonRatio

    @property
    @abstractmethod
    def default_threshold(self) -> Any:
        """Return the default threshold if an explicit one is missing."""

    def __init__(self, *args, threshold: Optional[Any] = None, **kwargs) -> None:
        """Init the ThresholdComparisonRatioCalculator."""
        super().__init__(*args, **kwargs)
        self._threshold = self.default_threshold if threshold is None else threshold

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        calc = self._calcs[0]
        assert isinstance(calc, MetricCalculator)
        peek = calc.peek

        # keep self.nan in results for nan values, that cannot be compared
        out = np.full(peek.shape, self.nan, dtype=np.int8)
        if np.isnan(calc.metric.nan):
            compare_where = ~np.isnan(peek)
        else:
            compare_where = peek != calc.metric.nan
        self._compare(calc.peek, self._threshold, out=out, where=compare_where)

        return out

    def _agg(self, samples: np.ndarray, offsets: np.ndarray) -> float:
        lenghts = np.diff(offsets.base)
        reduced_samples = np.add.reduceat(samples, offsets)
        # avoid zero division + reduceat copies the previous value
        vec = np.zeros(reduced_samples.shape, dtype=self.dtype)
        np.divide(reduced_samples, lenghts, where=lenghts != 0, out=vec)
        return vec

    def _compare(
        self,
        peek: np.ndarray,
        threshold: Any,
        *,
        out: np.ndarray,
        where: np.ndarray,
    ) -> np.ndarray:
        """Compare a metric value against the threshold."""
        raise NotImplementedError()


class HistogramCalculatorEnsemble(MetricCalculatorEnsemble):
    """Like MetricCalculatorEnsemble, but for histograms."""

    def __init__(
        self,
        *metrics: str,
        class_mapping: dict[str, Type[MetricCalculator]],
        quantiles: Sequence[float],
        **kwargs,
    ):
        """Initialize a new instance of HistogramCalculatorEnsemble class."""
        super().__init__(
            *metrics,
            class_mapping=class_mapping,
            quantiles=quantiles,
            quantile_stride=-1,
            **kwargs,
        )

    def _compose_quantile_time_intervals(
        self,
        min_time: np.datetime64,
        max_time: np.datetime64,
    ) -> tuple[np.ndarray, np.ndarray]:
        # matches the legacy outliers behavior - unstable, but suites histograms better
        return np.array([min_time]), np.array([max_time])

    def histograms(
        self,
        scale: Optional[Scale],
        bins: Optional[int],
        ticks: Optional[list],
    ) -> dict[str, Histogram]:
        """Calculate the current histograms."""
        return {k: v[0].histogram(scale, bins, ticks) for k, v in self._metrics.items()}


M = TypeVar("M", Metric, Histogram)


class BinnedEnsemblesCalculator(Generic[M]):
    """
    Common binned aggregations logic.

    We support multiple:
    - Time interval series.
    - Repository groups.
    - Metrics.
    """

    ensemble_class: Type[MetricCalculatorEnsemble]

    def __init__(
        self,
        metrics: Iterable[Sequence[str]],
        quantiles: Sequence[float],
        quantile_stride: int,
        **kwargs,
    ):
        """
        Initialize a new instance of `BinnedEnsemblesCalculator`.

        :param metrics: Series of series of metric names. Each inner series becomes a separate \
                        ensemble.
        :param quantiles: Pair of quantiles, common for each metric.
        """
        self.ensembles = [
            self.ensemble_class(
                *metrics,
                quantiles=quantiles,
                quantile_stride=quantile_stride,
                **kwargs,
            )
            for metrics in metrics
        ]
        self._metrics = list(metrics)

    @sentry_span
    def __call__(
        self,
        items: pd.DataFrame,
        time_intervals: Sequence[Sequence[datetime]],
        groups: np.ndarray,
        agg_kwargs: Iterable[Mapping[str, Any]],
    ) -> np.ndarray:
        """
        Calculate the binned aggregations on a series of mined facts.

        :param items: Calculate metrics on the elements of this DataFrame.
        :param time_intervals: Time interval borders in UTC. Each interval spans \
                               `[time_intervals[i], time_intervals[i + 1]]`, the ending is \
                               not included.
        :param groups: Calculate separate metrics for each group. Groups are represented by \
                       a multi-dimensional tensor with indexes to `items`.
        :param agg_kwargs: Keyword arguments to be passed to the ensemble aggregation.
        :return:   ensembles \
                 x groups \
                 x time intervals primary \
                 x time intervals secondary \
                 x metrics; \
                 3D numpy array of list[list[Metric]]] (dtype object).
        """
        assert isinstance(items, pd.DataFrame)
        assert isinstance(groups, np.ndarray)
        min_times, max_times, ts_index_map = self._make_min_max_times(time_intervals)
        if groups.dtype != object:
            assert np.issubdtype(groups.dtype, np.integer)
            flat_groups = groups.reshape(-1, groups.shape[-1])
            effective_groups_shape = groups.shape[:-1]
        else:
            flat_groups = groups.ravel()
            effective_groups_shape = groups.shape
        for ensemble in self.ensembles:
            ensemble(items, min_times, max_times, flat_groups)
        values_dicts = self._aggregate_ensembles(agg_kwargs)
        metrics = self._metrics

        def fill_ensemble_group(ensemble_index: int, group_index: int) -> np.ndarray:
            """Return Array[list[list[M]]]."""
            cell = np.full(len(time_intervals), None, object)
            cell[:] = [[[] for _ in range(len(ts) - 1)] for ts in time_intervals]
            values_dict = values_dicts[ensemble_index]
            for metric in metrics[ensemble_index]:
                for tix, value in enumerate(values_dict[metric][group_index]):
                    primary, secondary = ts_index_map[tix]
                    cell[primary][secondary].append(value)
            return cell

        flat_vals = np.concatenate(
            [
                fill_ensemble_group(m, g)
                for m in range(len(metrics))
                for g in range(flat_groups.shape[0])
            ],
        )
        return flat_vals.reshape((len(metrics), *effective_groups_shape, len(time_intervals)))

    @classmethod
    def _make_min_max_times(
        cls,
        time_intervals: Sequence[Sequence[datetime]],
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        sizes = np.zeros(len(time_intervals) + 1, dtype=int)
        for i, ts in enumerate(time_intervals):
            size = len(ts)
            sizes[i + 1] = size
            assert size >= 2, "Each time interval series must contain at least two elements."
        flat_time_intervals = np.fromiter(
            chain.from_iterable((dt.replace(tzinfo=None) for dt in ts) for ts in time_intervals),
            dtype="datetime64[ns]",
            count=sizes.sum(),
        )
        offsets = np.cumsum(sizes)
        flat_intervals_count = len(flat_time_intervals) - len(time_intervals)
        nat = np.datetime64("NaT")
        min_times = np.full_like(flat_time_intervals, nat, shape=flat_intervals_count)
        max_times = np.full_like(flat_time_intervals, nat, shape=flat_intervals_count)
        offset = 0
        ts_index_map = []
        for i in range(len(time_intervals)):
            begin, end = offsets[i], offsets[i + 1]
            next_offset = offset + end - begin - 1
            min_times[offset:next_offset] = flat_time_intervals[begin : end - 1]
            max_times[offset:next_offset] = flat_time_intervals[begin + 1 : end]
            for j in range(offset, next_offset):
                ts_index_map.append((i, j - offset))
            offset = next_offset
        return min_times, max_times, ts_index_map

    def _aggregate_ensembles(
        self,
        kwargs: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, list[list[M]]]]:
        raise NotImplementedError


class BinnedMetricCalculator(BinnedEnsemblesCalculator[Metric]):
    """Batched metrics calculation on sequential time intervals with always one set of metrics."""

    def __init__(
        self,
        metrics: Sequence[str],
        quantiles: Sequence[float],
        quantile_stride: int,
        **kwargs,
    ):
        """
        Initialize a new instance of `BinnedMetricsCalculator`.

        :param metrics: Sequence of metric names to calculate in each bin.
        :param quantiles: Pair of quantiles, common for each metric.
        :param quantile_stride: Size of the quantile locality in days.
        """
        super().__init__([metrics], quantiles=quantiles, quantile_stride=quantile_stride, **kwargs)

    def __call__(
        self,
        items: pd.DataFrame,
        time_intervals: Sequence[Sequence[datetime]],
        groups: np.ndarray,
    ) -> np.ndarray:
        """
        Override the parent's method to reduce the level of nesting.

        :return: array of list[list[Metric]]].
        """
        return super().__call__(items, time_intervals, groups, [{}])[0]

    def _aggregate_ensembles(
        self,
        kwargs: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, list[list[Metric]]]]:
        return [self.ensembles[0].values()]


class BinnedHistogramCalculator(BinnedEnsemblesCalculator[Histogram]):
    """Batched histograms calculation on sequential time intervals."""

    def __init__(self, metrics: Iterable[Sequence[str]], quantiles: Sequence[float], **kwargs):
        """Initialize a new instance of `BinnedHistogramCalculator`."""
        self.ensembles = [
            self.ensemble_class(
                *metrics,
                quantiles=quantiles,
                **kwargs,
            )
            for metrics in metrics
        ]
        self._metrics = list(metrics)

    def _aggregate_ensembles(
        self,
        kwargs: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, list[list[Histogram]]]]:
        return [
            {k: v for k, v in ensemble.histograms(**dataclass_asdict(ekw)).items()}
            for ensemble, ekw in zip(self.ensembles, kwargs)
        ]


def group_to_indexes(
    items: pd.DataFrame,
    *groupers: Callable[[pd.DataFrame], list[np.ndarray]],
    deduplicate_key: Optional[str] = None,
    deduplicate_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply a chain of grouping functions to a table and return the tensor with group indexes.

    :param deduplicate_key: Integer column name in `items` to scan for duplicates in each group.
    :param deduplicate_mask: Additional values to compare beside `deduplicate_key`.
    """
    if not groupers:
        return np.arange(len(items))[None, :]
    groups = [grouper(items) for grouper in groupers]
    boilerplate = np.empty(
        len(items),
        dtype=np.uint8
        if len(groups) < (2 << 8)
        else np.uint16
        if len(groups) < (1 << 16)
        else np.uint32,
    )

    def intersect(*coordinates: int) -> np.ndarray:
        boilerplate[:] = 0
        for group, i in zip(groups, coordinates):
            boilerplate[group[i]] += 1
        return np.flatnonzero(boilerplate == len(groups))

    indexes = np.fromfunction(
        np.vectorize(intersect, otypes=[object]), [len(g) for g in groups], dtype=object,
    )
    return deduplicate_groups(indexes, items, deduplicate_key, deduplicate_mask)


def deduplicate_groups(
    indexes: np.ndarray,
    items: pd.DataFrame,
    deduplicate_key: Optional[str],
    deduplicate_mask: Optional[np.ndarray],
) -> np.ndarray:
    """
    Deduplicate logical items in groups of `items`.

    :param deduplicate_key: Integer column name in `items` to scan for duplicates in each group.
    :param deduplicate_mask: Additional values to compare beside `deduplicate_key`.
    """
    if deduplicate_key is None:
        return indexes
    deduped_indexes = np.empty_like(indexes)
    deduped_indexes_flat = deduped_indexes.ravel()
    key_arr = items[deduplicate_key].values
    if deduplicate_mask is not None and len(key_arr):
        key_arr = int_to_str(key_arr, deduplicate_mask)
    for i, group in enumerate(indexes.ravel()):
        _, group_indexes = np.unique(key_arr[group], return_index=True)
        if len(group_indexes) < len(group):
            group_indexes = group[group_indexes]
        else:
            group_indexes = group
        deduped_indexes_flat[i] = group_indexes
    return deduped_indexes


def group_by_repo(
    repository_full_name_column_name: str,
    repos: Sequence[Collection[str]],
    df: pd.DataFrame,
) -> list[np.ndarray]:
    """Group items by the value of their "repository_full_name" column."""
    if df.empty:
        return [np.array([], dtype=int)] * len(repos)
    df_repos = df[repository_full_name_column_name].values.astype("S")
    repos = [
        np.array(
            repo_group if not isinstance(repo_group, (set, KeysView)) else list(repo_group),
            dtype="S",
        )
        for repo_group in repos
    ]
    unique_repos, imap = np.unique(np.concatenate(repos), return_inverse=True)
    if len(unique_repos) <= len(repos):
        matches = np.array([df_repos == repo for repo in unique_repos])
        pos = 0
        result = []
        for repo_group in repos:
            step = len(repo_group)
            cols = imap[pos : pos + step]
            group = np.flatnonzero(np.sum(matches[cols], axis=0, dtype=bool))
            pos += step
            result.append(group)
    else:
        result = [np.flatnonzero(np.in1d(df_repos, repo_group)) for repo_group in repos]
    return result


def group_by_lines(lines: Sequence[int], column: np.ndarray) -> list[np.ndarray]:
    """
    Bin items by the number of changed `lines` represented by `column`.

    We throw away the ends: PRs with fewer lines than `lines[0]` and with more lines than \
    `lines[-1]`.

    :param lines: Either an empty sequence or one with at least 2 elements. The numbers must \
                  monotonically increase.
    """
    lines = np.asarray(lines)
    if len(lines) and len(column):
        assert len(lines) >= 2
        assert (np.diff(lines) > 0).all()
    else:
        return [np.arange(len(column))]
    line_group_assignments = np.digitize(column, lines)
    line_group_assignments[line_group_assignments == len(lines)] = 0
    line_group_assignments -= 1
    order = np.argsort(line_group_assignments)
    existing_groups, existing_group_counts = np.unique(
        line_group_assignments[order], return_counts=True,
    )
    line_groups = np.split(np.arange(len(column))[order], np.cumsum(existing_group_counts)[:-1])
    if line_group_assignments[order[0]] < 0:
        line_groups = line_groups[1:]
        existing_groups = existing_groups[1:]
    full_line_groups = [np.array([], dtype=int)] * (len(lines) - 1)
    for i, g in zip(existing_groups, line_groups):
        full_line_groups[i] = g
    return full_line_groups


@dataclasses.dataclass(frozen=True, slots=True)
class JIRAGrouping:
    """Grouping definition based on jira entities.

    `None` field values indicate that no grouping should be done on the corresponding dataframe
    property.
    On the opposite, a single empty field value makes this group select nothing.
    """

    projects: Optional[Collection[str]] = None
    priorities: Optional[Collection[str]] = None
    types: Optional[Collection[str]] = None
    labels: Optional[Collection[str]] = None

    def __bool__(self) -> bool:
        """Get whether the group definition is empty.  An empty group does not do any grouping."""
        values = (self.projects, self.priorities, self.types, self.labels)
        return any(val is not None for val in values)

    @classmethod
    def empty(cls) -> JIRAGrouping:
        """Return a `JIRAGrouping` selecting all data."""
        return cls()


def group_pr_facts_by_jira(
    jira_groups: Sequence[JIRAGrouping],
    df: pd.DataFrame,
) -> list[np.ndarray]:
    """Build the groups according to the JIRA information expressed by the `jira_groups`.

    `df` is the dataframe built from `PullRequestFacts` structs.

    """
    if df.empty:
        return [np.array([], dtype=int)] * len(jira_groups)
    res = []
    arrays_matchers: dict[str, _ArraysWhitelistMatcher] = {}  # each prop will have a wl matcher
    for jira_group in jira_groups:
        if jira_group:
            df_matches = np.zeros(len(df), dtype=np.uint8)
            required_n_matches = 0

            DETAILS = PR_JIRA_DETAILS_COLUMN_MAP
            for group_props, df_col, dtype in (
                (jira_group.projects, "jira_projects", DETAILS[Issue.project_id][1]),
                (jira_group.priorities, "jira_priorities", DETAILS[Issue.priority_id][1]),
                (jira_group.types, "jira_types", DETAILS[Issue.type_id][1]),
                (jira_group.labels, "jira_labels", DETAILS[Issue.labels][1]),
            ):
                if group_props is not None:
                    if (matcher := arrays_matchers.get(df_col)) is None:
                        arrays_matchers[df_col] = matcher = _ArraysWhitelistMatcher(df[df_col])
                    required_n_matches += 1
                    filter_values = np.array(list(group_props), dtype=dtype)
                    field_res = matcher(filter_values)
                    df_matches[field_res] += 1

            group_res = np.flatnonzero(df_matches == required_n_matches)
        else:
            group_res = np.arange(len(df), dtype=int)

        res.append(group_res)
    return res


group_release_facts_by_jira = group_pr_facts_by_jira
"""Build the groups according to the JIRA information expressed by the `jira_groups`.

`df` is the dataframe built from `ReleaseFacts` structs.

"""


def group_jira_facts_by_jira(
    jira_groups: Sequence[JIRAGrouping],
    df: pd.Dataframe,
) -> list[np.ndarray]:
    """Build the groups according to `jira_groups`.

    `df` is the dataframe in the format returned by `fetch_jira_issues`, built from Issue table.
    """
    if df.empty:
        return [np.array([], dtype=int)] * len(jira_groups)
    res = []

    labels_matcher: _ArraysWhitelistMatcher | None = None

    for jira_group in jira_groups:
        required_n_matches = 0
        if jira_group:
            DETAILS = PR_JIRA_DETAILS_COLUMN_MAP

            matches = np.zeros(len(df), dtype=np.uint8)
            for group_props, df_col, dtype in (
                (jira_group.projects, Issue.project_id.name, DETAILS[Issue.project_id][1]),
                (jira_group.priorities, Issue.priority_id.name, DETAILS[Issue.priority_id][1]),
                (jira_group.types, Issue.type_id.name, DETAILS[Issue.type_id][1]),
                (jira_group.labels, Issue.labels.name, DETAILS[Issue.labels][1]),
            ):
                if group_props is not None:
                    # lazy access the dataframe column since it could be missing
                    # when no grouping is needed for the property
                    values = df[df_col].values
                    required_n_matches += 1
                    filter_values = np.array(list(group_props), dtype=dtype)
                    if df_col == Issue.labels.name:
                        if labels_matcher is None:
                            labels_matcher = _ArraysWhitelistMatcher(values)
                        matches[labels_matcher(filter_values)] += 1
                    else:
                        matches[in1d_str(values, filter_values)] += 1
                group_res = np.flatnonzero(matches == required_n_matches)
        else:
            group_res = np.arange(len(df), dtype=int)
        res.append(group_res)
    return res


class _ArraysWhitelistMatcher:
    """Search an array of arrays to find those having a value in the whitelist.

    The arrays of arrays is passed in the constructor then it can be matched against multiple
    whitelists.
    """

    def __init__(self, arrays: npt.NDArray[np.ndarray]):
        self._arrays = arrays
        self._splits = None
        self._all_values = None

    def __call__(self, whitelist: np.ndarray) -> npt.NDArray[int]:
        """Get indexes in `arrays` where the intersection with `whitelist` is not empty."""
        if self._splits is None:
            self._all_values = np.concatenate(self._arrays)
            self._splits = np.cumsum(
                np.fromiter((len(v) for v in self._arrays), int, len(self._arrays)),
            )
        match_mask = in1d_str(self._all_values, whitelist)
        match_indexes = np.flatnonzero(match_mask)
        return np.unique(np.searchsorted(self._splits, match_indexes, side="right"))


def calculate_logical_duplication_mask(
    repos_column: np.ndarray,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
) -> Optional[np.ndarray]:
    """
    Assign indexes to same logical settings for each logical repository.

    Each logical repository with the same logical settings (releases, deployments) gets
    assigned to the same index.
    Physical repos and unique logicals get index 0.
    """
    # maximum 255 logical releases for each repo -> uint8
    mask = np.zeros(len(repos_column), dtype=np.uint8)
    actual_unique_repos = unordered_unique(repos_column).astype("U")
    for root, children in coerce_logical_repos(actual_unique_repos).items():
        if len(children) == 1:
            continue
        groups = defaultdict(list)
        counter = 1
        try:
            deployments = logical_settings.deployments(root)
        except KeyError:
            deployments = None
        for child in children:
            child_releases = release_settings.native[child]
            if deployments is None:
                child_deployments = ("", ())
            else:
                try:
                    dep_re = deployments.title(child).pattern
                except KeyError:
                    dep_re = ""
                try:
                    dep_labels = tuple(
                        sorted((k, tuple(v)) for k, v in deployments.labels(child).items()),
                    )
                except KeyError:
                    dep_labels = ()
                child_deployments = dep_re, dep_labels
            child_key = (child_releases, child_deployments)
            groups[child_key].append(child)
        for group in groups.values():
            mask[np.in1d(repos_column, np.array(group, dtype="S"))] = counter
            counter += 1
    return mask


class RatioCalculator(WithoutQuantilesMixin, MetricCalculator[float]):
    """Calculate the ratio of two counts from the dependencies."""

    metric = MetricFloat
    value_offset = 0

    def __init__(self, *deps: MetricCalculator, quantiles: Sequence[float], **kwargs):
        """Initialize a new instance of RatioCalculator."""
        super().__init__(*deps, quantiles=quantiles, **kwargs)
        if isinstance(self._calcs[1], self.deps[0]):
            self._calcs = list(reversed(self._calcs))
        self._opened, self._closed = self._calcs

    def _values(self) -> list[list[Metric[float]]]:
        metrics = [
            [self.metric.from_fields(False, None, None, None)] * len(samples)
            for samples in self.samples
        ]
        offset = self.value_offset
        for i, (opened_group, closed_group) in enumerate(
            zip(self._opened.values, self._closed.values),
        ):
            for j, (opened, closed) in enumerate(zip(opened_group, closed_group)):
                if (not closed.exists and not opened.exists) or (
                    opened.value == closed.value == 0 and offset == 0
                ):
                    continue
                # offset may be 1, See ENG-866
                val = ((opened.value or 0) + offset) / ((closed.value or 0) + offset)
                metrics[i][j] = self.metric.from_fields(True, val, None, None)
        return metrics

    def _value(self, samples: np.ndarray) -> Metric[timedelta]:
        raise AssertionError("this must be never called")

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return np.full((len(min_times), len(facts)), self.nan, self.dtype)


def make_register_metric(
    metric_calculators: dict[str, Type[MetricCalculator]],
    histogram_calculators: Optional[dict[str, Type[HistogramCalculator]]],
):
    """Create the decorator to keep track of the metric and histogram calculators."""

    def register_metric(name: str):
        assert isinstance(name, str)

        def register_with_name(cls: Type[MetricCalculator]):
            metric_calculators[name] = cls
            if histogram_calculators is not None and not issubclass(cls, SumMetricCalculator):
                histogram_calculators[name] = type(
                    "HistogramOf" + cls.__name__, (cls, HistogramCalculator), {},
                )
            return cls

        return register_with_name

    return register_metric
