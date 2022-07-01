from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from functools import reduce
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
)
from athenian.api.internal.features.statistics import (
    mean_confidence_interval,
    median_confidence_interval,
)
from athenian.api.internal.logical_repos import coerce_logical_repos
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings
from athenian.api.sparse_mask import SparseMask
from athenian.api.tracing import sentry_span
from athenian.api.unordered_unique import unordered_unique

DEFAULT_QUANTILE_STRIDE = 14


class MetricCalculator(Generic[T], ABC):
    """
    Arbitrary type T metric calculator, base abstract class.

    Each call to `MetricCalculator()` feeds another input object to update the state.
    `MetricCalculator.value()` returns the current metric value.
    `MetricCalculator._analyze()` is to be implemented by the particular metric calculators.
    """

    # Types of dependencies - upstream MetricCalculator-s.
    deps: tuple[Type["MetricCalculator"], ...] = tuple()

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
        if self._quantiles != (0, 1):
            assert quantiles_mounted_at is not None
        if quantiles_mounted_at is not None:
            assert len(min_times) > quantiles_mounted_at

        assert isinstance(groups_mask, np.ndarray)
        assert len(groups_mask.shape) == 2
        assert len(groups_mask) > 0 or facts.empty
        if facts.empty:
            self._peek = np.empty((len(min_times), 0), object)
            ts_dim = quantiles_mounted_at or len(min_times)
            self._grouped_notnull = self._grouped_sample_mask = SparseMask(
                np.empty((len(groups_mask), ts_dim, 0), dtype=bool),
            )
            self._samples = np.empty((len(groups_mask), ts_dim, 0), self.dtype)
            return
        assert groups_mask.shape[1] == len(facts)
        self._peek = peek = self._analyze(facts, min_times, max_times, **kwargs)
        assert isinstance(peek, np.ndarray), type(self)
        if self.is_pure_dependency:
            return
        assert peek.shape == (len(min_times), len(facts)), (peek.shape, type(self))
        if self._quantiles != (0, 1):
            discard_mask = self._calculate_discard_mask(peek, quantiles_mounted_at, groups_mask)
        if quantiles_mounted_at is not None:
            peek = peek[:quantiles_mounted_at]
        notnull = self._find_notnull(peek)
        notnull = np.broadcast_to(notnull[None, :], (len(groups_mask), *notnull.shape))
        group_sample_mask = notnull & groups_mask[:, None, :]
        del notnull
        self._grouped_notnull = SparseMask(group_sample_mask)
        if self._quantiles != (0, 1):
            group_sample_mask = group_sample_mask.copy()
            discard_mask = np.broadcast_to(discard_mask[:, None, :], group_sample_mask.shape)
            group_sample_mask[discard_mask] = False
            del discard_mask
        self._grouped_sample_mask = SparseMask(group_sample_mask)
        flat_samples = (
            np.broadcast_to(peek[None, :], group_sample_mask.shape)[group_sample_mask]
            .astype(self.dtype, copy=False)
            .ravel()
        )
        group_sample_lengths = np.sum(group_sample_mask, axis=-1)
        self._samples = np.full(len(groups_mask) * len(peek), None, object)
        self._samples[:] = np.split(flat_samples, np.cumsum(group_sample_lengths.ravel())[:-1])
        self._samples = self._samples.reshape((len(groups_mask), len(peek)))

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
        self._grouped_notnull = self._grouped_sample_mask = SparseMask(
            np.empty((0, 0, 0), dtype=bool),
        )
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
    def _find_notnull(cls, peek: np.ndarray) -> np.ndarray:
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
        groups_mask: np.ndarray,
    ) -> np.ndarray:
        qnotnull = self._find_notnull(peek[quantiles_mounted_at:])
        if peek.dtype != object:
            dtype = peek.dtype
        else:
            dtype = self.dtype
        qnotnull = np.broadcast_to(qnotnull[None, :], (len(groups_mask), *qnotnull.shape))
        group_sample_mask = qnotnull & groups_mask[:, None, :]
        flat_samples = (
            np.broadcast_to(peek[None, quantiles_mounted_at:], qnotnull.shape)[group_sample_mask]
            .astype(dtype, copy=False)
            .ravel()
        )
        group_sample_lengths = np.sum(group_sample_mask, axis=-1)
        if self.has_nan:
            max_len = group_sample_lengths.max()
            if max_len == 0:
                cut_values = np.full((*group_sample_lengths.shape, 2), None, dtype)
            else:
                surrogate_samples = np.full((*group_sample_lengths.shape, max_len), None, dtype)
                flat_lengths = group_sample_lengths.ravel()
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
                            surrogate_samples, self._quantiles, interpolation="nearest", axis=-1,
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
                        flat_samples[pos : pos + slen], self._quantiles, interpolation="nearest",
                    )
                    pos += slen
        cut_values = cut_values.reshape((*group_sample_lengths.shape, 2))
        grouped_qpeek = np.broadcast_to(peek[None, quantiles_mounted_at:], qnotnull.shape)
        group_qmask = np.broadcast_to(groups_mask[:, None, :], grouped_qpeek.shape)
        if self._quantiles[0] > 0:
            discard_left_mask = np.less(
                grouped_qpeek, cut_values[:, :, 0][:, :, None], where=group_qmask,
            ).any(axis=1)
        if self._quantiles[1] < 1:
            discard_right_mask = np.greater(
                grouped_qpeek, cut_values[:, :, 1][:, :, None], where=group_qmask,
            ).any(axis=1)
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

    def _value(self, samples: np.ndarray) -> Metric[T]:
        exists = len(samples) > 0
        return self.metric.from_fields(
            True, self._agg(samples) if exists else np.dtype(self.dtype).type(), None, None,
        )

    @abstractmethod
    def _agg(self, samples: np.ndarray) -> T:
        raise NotImplementedError


class SumMetricCalculator(AggregationMetricCalculator[T], ABC):
    """Sum calculator."""

    def _agg(self, samples: np.ndarray) -> T:
        return samples.sum()


class MaxMetricCalculator(AggregationMetricCalculator[T], ABC):
    """Maximum calculator."""

    def _agg(self, samples: np.ndarray) -> T:
        return samples.max()


class AnyMetricCalculator(AggregationMetricCalculator[T], ABC):
    """len(samples) > 0 calculator."""

    def _agg(self, samples: np.ndarray) -> T:
        return len(samples) > 0


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
            {k: v for k, v in ensemble.histograms(**ekw).items()}
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

    def intersect(*coordinates: int) -> np.ndarray:
        return reduce(
            lambda x, y: np.intersect1d(x, y, assume_unique=True),
            [group[i] for group, i in zip(groups, coordinates)],
        )

    indexes = np.fromfunction(
        np.vectorize(intersect, otypes=[object]), [len(g) for g in groups], dtype=object,
    )
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
