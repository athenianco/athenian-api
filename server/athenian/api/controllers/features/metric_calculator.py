from datetime import datetime, timedelta
from itertools import chain
from typing import Any, Collection, Dict, Generic, Iterable, List, Optional, Sequence, \
    Tuple, Type, TypeVar

import networkx as nx
import numpy as np
import pandas as pd
from pandas._libs import tslib

from athenian.api import typing_utils
from athenian.api.controllers.features.histogram import calculate_histogram, Histogram, Scale
from athenian.api.controllers.features.metric import Metric, T
from athenian.api.controllers.features.statistics import mean_confidence_interval, \
    median_confidence_interval
from athenian.api.tracing import sentry_span


class MetricCalculator(Generic[T]):
    """
    Arbitrary type T metric calculator, base abstract class.

    Each call to `MetricCalculator()` feeds another input object to update the state.
    `MetricCalculator.value()` returns the current metric value.
    `MetricCalculator._analyze()` is to be implemented by the particular metric calculators.
    """

    # Types of dependencies - upstream MetricCalculator-s.
    deps = tuple()

    # numpy data type of the metric value
    dtype = None

    def __init__(self, *deps: "MetricCalculator", quantiles: Sequence[float]):
        """Initialize a new `MetricCalculator` instance."""
        self.reset()
        self._calcs = []
        self._quantiles = np.asarray(quantiles)
        for calc in deps:
            for cls in self.deps:
                if isinstance(calc, cls):
                    self._calcs.append(calc)
                    break
            else:
                raise AssertionError("%s is not listed in the dependencies" % type(calc))

    def __call__(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 groups: Optional[Sequence[Sequence[int]]],
                 **kwargs) -> None:
        """Calculate the samples over the supplied facts.

        :param facts: Mined facts about pull requests: timestamps, stats, etc.
        :param min_times: Beginnings of the considered time intervals. They are needed to discard \
                          samples with both ends less than the minimum time.
        :param max_times: Endings of the considered time intervals. They are needed to discard \
                          samples with both ends greater than the maximum time.
        :param groups: Series of indexes to split the samples into.
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
        assert len(groups) > 0
        if facts.empty:
            self._peek = np.empty((len(min_times), 0), object)
            self._samples = [[np.array([], self.dtype)
                              for _ in range(len(min_times))] for _ in groups]
            return
        self._peek = peek = self._analyze(facts, min_times, max_times, **kwargs)
        assert isinstance(peek, np.ndarray), type(self)
        assert peek.shape[:2] == (len(min_times), len(facts)), (peek.shape, type(self))
        notnull = peek != np.array(None)
        gpeek = [peek[:, g] for g in groups]
        gnotnull = [notnull[:, g] for g in groups]
        self._samples = [[p[nn].astype(self.dtype)
                          for p, nn in zip(gp, gnn)]
                         for gp, gnn in zip(gpeek, gnotnull)]

    @property
    def values(self) -> List[List[Metric[T]]]:
        """Return the current metric value."""
        if self._last_values is None:
            self._last_values = self._values()
        return self._last_values

    @property
    def peek(self) -> np.ndarray:
        """Return the last calculated samples, with None-s."""
        return self._peek

    @property
    def samples(self) -> List[List[np.ndarray]]:
        """Return the last calculated samples, without None-s: groups x time intervals x facts."""
        return self._samples

    def reset(self) -> None:
        """Clear the current state of the calculator."""
        self._samples = []  # type: List[List[np.ndarray]]
        self._peek = np.empty((0, 0), dtype=object)
        self._last_values = None

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        """Calculate the samples for each item in the data frame."""
        raise NotImplementedError

    def _value(self, samples: np.ndarray) -> Metric[T]:
        """Calculate the metric values from the current samples."""
        raise NotImplementedError

    def _values(self) -> List[List[Metric[T]]]:
        return [[self._value(self._cut_by_quantiles(s)) for s in gs] for gs in self._samples]

    def _cut_by_quantiles(self, samples: np.ndarray) -> np.ndarray:
        """Cut from the left and the right of the distribution by quantiles."""
        if len(samples) == 0 or self._quantiles[0] == 0 and self._quantiles[1] == 1:
            return samples
        cut_values = np.quantile(samples, self._quantiles)
        if self._quantiles[0] != 0:
            samples = np.delete(samples, np.where(samples < cut_values[0])[0])
        samples = np.delete(samples, np.where(samples > cut_values[1])[0])
        return samples


class AverageMetricCalculator(MetricCalculator[T]):
    """Mean calculator."""

    may_have_negative_values: bool

    def _value(self, samples: np.ndarray) -> Metric[T]:
        if len(samples) == 0:
            return Metric(False, None, None, None)
        assert self.may_have_negative_values is not None
        if not self.may_have_negative_values:
            zero = samples.dtype.type(0)
            negative = np.where(samples < zero)[0]
            assert len(negative) == 0, samples[negative]
        return Metric(True, *mean_confidence_interval(samples, self.may_have_negative_values))

    def _analyze(self, facts: Any, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[T]:
        raise NotImplementedError


class MedianMetricCalculator(MetricCalculator[T]):
    """Median calculator."""

    def _value(self, samples: np.ndarray) -> Metric[T]:
        if len(samples) == 0:
            return Metric(False, None, None, None)
        return Metric(True, *median_confidence_interval(samples))

    def _analyze(self, facts: Any, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[T]:
        raise NotImplementedError


class SumMetricCalculator(MetricCalculator[T]):
    """Sum calculator."""

    def _value(self, samples: np.ndarray) -> Metric[T]:
        exists = len(samples) > 0
        return Metric(True, samples.sum() if exists else np.dtype(self.dtype).type(), None, None)

    def _analyze(self, facts: Any, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[T]:
        raise NotImplementedError


class Counter(MetricCalculator[int]):
    """Count the number of PRs that were used to calculate the specified metric."""

    dtype = int

    def _value(self, samples: np.ndarray) -> Metric[int]:
        return Metric(True, len(samples), None, None)

    def __call__(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 groups: Optional[Sequence[Sequence[int]]],
                 **kwargs) -> None:
        """Reference the same peek and samples from the only dependency."""  # noqa
        self._peek = self._calcs[0].peek
        self._samples = self._calcs[0].samples


class WithoutQuantilesMixin:
    """Ignore the quantiles."""

    def _cut_by_quantiles(self, samples: np.ndarray) -> np.ndarray:
        return samples


class HistogramCalculator(MetricCalculator):
    """Pull request histogram calculator, base abstract class."""

    def histogram(self,
                  scale: Optional[Scale],
                  bins: Optional[int],
                  ticks: Optional[list],
                  ) -> List[List[Histogram[T]]]:
        """Calculate the histogram over the current distribution."""
        histograms = []
        for group_samples in self.samples:
            histograms.append(group_histograms := [])
            for samples in group_samples:
                samples = self._cut_by_quantiles(samples)
                if scale == Scale.LOG:
                    if (shift_log := getattr(self, "_shift_log", None)) is not None:
                        samples = shift_log(samples)
                group_histograms.append(calculate_histogram(samples, scale, bins, ticks))
        return histograms


class MetricCalculatorEnsemble:
    """Several calculators ordered in sequence such that the dependencies are respected and each \
    calculator class is calculated only once."""

    def __init__(self,
                 *metrics: str,
                 class_mapping: Dict[str, Type[MetricCalculator]],
                 quantiles: Sequence[float]):
        """Initialize a new instance of MetricCalculatorEnsemble class."""
        metric_classes = {class_mapping[m]: m for m in metrics}
        self._calcs, self._metrics = self._plan_classes(metric_classes, quantiles)

    @staticmethod
    def _plan_classes(metric_classes: Dict[Type[MetricCalculator], str],
                      quantiles: Sequence[float],
                      ) -> Tuple[List[MetricCalculator],
                                 Dict[str, MetricCalculator]]:
        dig = nx.DiGraph()
        required_classes = list(metric_classes)
        while required_classes:
            cls = required_classes.pop()
            if cls.deps:
                for dep in cls.deps:
                    if dep not in dig:
                        required_classes.append(dep)
                    dig.add_edge(cls, dep)
            elif cls not in dig:
                dig.add_node(cls)
        calcs = []
        metrics = {}
        cls_instances = {}
        for cls in reversed(list(nx.topological_sort(dig))):
            calc = cls(*[cls_instances[dep] for dep in cls.deps],
                       quantiles=quantiles)
            calcs.append(calc)
            cls_instances[cls] = calc
            try:
                metrics[metric_classes[cls]] = calc
            except KeyError:
                continue
        return calcs, metrics

    @sentry_span
    def __call__(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 groups: Optional[Sequence[Sequence[int]]],
                 **kwargs) -> None:
        """Invoke all the owned metric calculators on the same input."""
        for calc in self._calcs:
            calc(facts, min_times, max_times, groups, **kwargs)

    def __bool__(self) -> bool:
        """Return True if there is at leat one calculator inside; otherwise, False."""
        return bool(len(self._calcs))

    @sentry_span
    def values(self) -> Dict[str, List[List[Metric[T]]]]:
        """Calculate the current metric values."""
        return {k: v.values for k, v in self._metrics.items()}

    def reset(self) -> None:
        """Clear the states of all the contained calculators."""
        [c.reset() for c in self._calcs]


class HistogramCalculatorEnsemble(MetricCalculatorEnsemble):
    """Like MetricCalculatorEnsemble, but for histograms."""

    def histograms(self,
                   scale: Optional[Scale],
                   bins: Optional[int],
                   ticks: Optional[list],
                   ) -> Dict[str, Histogram]:
        """Calculate the current histograms."""
        return {k: v.histogram(scale, bins, ticks) for k, v in self._metrics.items()}


@sentry_span
def df_from_dataclasses(items: Iterable[Any], length: Optional[int] = None) -> pd.DataFrame:
    """Combine several dataclasses to a Pandas DataFrame."""
    columns = {}
    first_item = None
    try:
        if length is None:
            length = len(items)
    except TypeError:
        # slower branch without pre-allocation
        for i, item in enumerate(items):
            if i == 0:
                first_item = item
                for k in item.__dict__:
                    columns[k] = []
            for k, v in item.__dict__.items():
                columns[k].append(v)
    else:
        for i, item in enumerate(items):
            if i == 0:
                first_item = item
                for k in item.__dict__:
                    columns[k] = [None] * length
            # dataclasses.asdict() creates a new dict and is too slow
            for k, v in item.__dict__.items():
                columns[k][i] = v
    if first_item is None:
        return pd.DataFrame()
    column_types = {}
    for k, v in type(first_item).__annotations__.items():
        if typing_utils.is_optional(v):
            v = v.__args__[0]
        elif typing_utils.is_generic(v):
            v = object
        column_types[k] = v
    for k, v in columns.items():
        column_type = column_types[k]
        if issubclass(column_type, datetime):
            v = tslib.array_to_datetime(np.array(v, dtype=object), utc=True, errors="raise")[0]
        elif issubclass(column_type, timedelta):
            v = np.array(v, dtype="timedelta64[s]")
        elif np.dtype(column_type) != np.dtype(object):
            v = np.array(v, dtype=column_type)
        columns[k] = v
    df = pd.DataFrame.from_dict(columns)
    return df


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

    def __init__(self,
                 metrics: Sequence[Sequence[str]],
                 quantiles: Sequence[float],
                 **kwargs):
        """
        Initialize a new instance of `BinnedMetricCalculator`.

        :param metrics: Series of series of metric names. Each inner series becomes a separate \
                        ensemble.
        :param quantiles: Pair of quantiles, common for each metric.
        """
        self.ensembles = [
            self.ensemble_class(*metrics, quantiles=quantiles, **kwargs) for metrics in metrics
        ]
        self.metrics = metrics

    @sentry_span
    def __call__(self,
                 items: Dict[str, Iterable[Any]],
                 time_intervals: Sequence[Sequence[datetime]],
                 groups: Sequence[Collection[str]],
                 kwargs: Iterable[Dict[str, Any]],
                 ) -> List[List[List[List[List[M]]]]]:
        """
        Calculate the binned aggregations on a series of mined facts.

        :param items: Map from repository names to their facts.
        :param time_intervals: Time interval borders in UTC. Each interval spans \
                               `[time_intervals[i], time_intervals[i + 1]]`, the ending \
                               not included.
        :param groups: Repositories within the same group share the metrics scope.
        :return:   ensembles \
                 x time intervals primary \
                 x splitted groups (groups x splits) \
                 x time intervals secondary \
                 x metrics.
        """
        sizes = [len(repo_items) for repo_items in chain([[]], items.values())]
        df_items = df_from_dataclasses(chain.from_iterable(items.values()), length=sum(sizes))
        splits, max_splits = self._split_items(df_items)
        offsets = np.cumsum(sizes)
        keyix = {k: i for i, k in enumerate(items)}
        group_indexes = []
        for group in groups:
            indexes = []
            for k in group:
                try:
                    indexes.append(np.arange(offsets[keyix[k]], offsets[keyix[k] + 1]))
                except KeyError:
                    continue
            indexes = np.concatenate(indexes) if indexes else np.array([], dtype=int)
            if max_splits > 1:
                group_splits = splits[indexes]
                order = np.argsort(group_splits)
                indexes = indexes[order]
                group_splits = group_splits[order]
                split_indexes, split_counts = np.unique(group_splits, return_counts=True)
                if split_indexes[0] < 0:
                    # remove excluded items
                    indexes = indexes[split_counts[0]:]
                    split_counts = split_counts[1:]
                    split_indexes = split_indexes[1:]
                indexes = dict(zip(split_indexes, np.split(indexes, np.cumsum(split_counts)[:-1])))
            else:
                indexes = {0: indexes}
            for sx in range(max_splits):
                group_indexes.append(indexes.get(sx, []))

        min_times, max_times, ts_index_map = self._make_min_max_times(time_intervals)

        for ensemble in self.ensembles:
            ensemble(df_items, min_times, max_times, group_indexes)

        values_dicts = self._aggregate_ensembles(kwargs)
        result = [[[[[None] * len(metrics)
                     for _ in range(len(ts) - 1)]
                    for _ in range(len(groups) * max_splits)]
                   for ts in time_intervals]
                  for metrics in self.metrics]
        for eix, (metrics, values_dict) in enumerate(zip(self.metrics, values_dicts)):
            for mix, metric in enumerate(metrics):
                for gix, group in enumerate(values_dict[metric]):
                    for tix, value in enumerate(group):
                        primary, secondary = ts_index_map[tix]
                        result[eix][primary][gix][secondary][mix] = value
        return result

    def _split_items(self, items: pd.DataFrame) -> Tuple[np.ndarray, int]:
        return np.zeros(len(items)), 1

    @classmethod
    def _make_min_max_times(cls, time_intervals: Sequence[Sequence[datetime]],
                            ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        sizes = np.zeros(len(time_intervals) + 1, dtype=int)
        for i, ts in enumerate(time_intervals):
            size = len(ts)
            sizes[i + 1] = size
            assert size >= 2, "Each time interval series must contain at least two elements."
        flat_time_intervals = np.fromiter(
            chain.from_iterable((dt.replace(tzinfo=None) for dt in ts) for ts in time_intervals),
            dtype="datetime64[ns]", count=sizes.sum())
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
            min_times[offset:next_offset] = flat_time_intervals[begin:end - 1]
            max_times[offset:next_offset] = flat_time_intervals[begin + 1:end]
            for j in range(offset, next_offset):
                ts_index_map.append((i, j - offset))
            offset = next_offset
        return min_times, max_times, ts_index_map

    def _aggregate_ensembles(self, kwargs: Iterable[Dict[str, Any]],
                             ) -> List[Dict[str, List[List[M]]]]:
        raise NotImplementedError


class BinnedMetricsCalculator(BinnedEnsemblesCalculator[Metric]):
    """Batched metrics calculation on sequential time intervals with always one set of metrics."""

    def __init__(self,
                 metrics: Sequence[str],
                 quantiles: Sequence[float],
                 **kwargs):
        """
        Initialize a new instance of `BinnedMetricsCalculator`.

        :param metrics: Sequence of metric names to calculate in each bin.
        :param quantiles: Pair of quantiles, common for each metric.
        """
        super().__init__([metrics], quantiles, **kwargs)

    def __call__(self,
                 items: Dict[str, Iterable[Any]],
                 time_intervals: Sequence[Sequence[datetime]],
                 groups: Sequence[Collection[str]],
                 ) -> List[List[List[List[Metric]]]]:
        """Override the parent's method to reduce the number of nested lists."""
        return super().__call__(items, time_intervals, groups, [{}])[0]

    def _aggregate_ensembles(self, kwargs: Iterable[Dict[str, Any]],
                             ) -> List[Dict[str, List[List[Metric]]]]:
        return [self.ensembles[0].values()]
