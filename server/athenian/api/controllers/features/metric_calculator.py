from datetime import datetime, timedelta
from functools import reduce
from itertools import chain
from typing import Any, Callable, Collection, Dict, Generic, Iterable, List, Mapping, Optional, \
    Sequence, \
    Tuple, Type, TypeVar

import networkx as nx
import numpy as np
import pandas as pd
from pandas._libs import tslib
import sentry_sdk

from athenian.api import typing_utils
from athenian.api.controllers.features.histogram import calculate_histogram, Histogram, Scale
from athenian.api.controllers.features.metric import Metric, T
from athenian.api.controllers.features.statistics import mean_confidence_interval, \
    median_confidence_interval
from athenian.api.controllers.miners.github.dag_accelerated import searchsorted_inrange
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
                 groups: Sequence[Sequence[int]],
                 **kwargs) -> None:
        """Calculate the samples over the supplied facts.

        :param facts: Mined facts about pull requests: timestamps, stats, etc.
        :param min_times: Beginnings of the considered time intervals. They are needed to discard \
                          samples with both ends less than the minimum time.
        :param max_times: Endings of the considered time intervals. They are needed to discard \
                          samples with both ends greater than the maximum time.
        :param groups: Series of group indexes to split the samples into.
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
        groups = np.asarray(groups)
        assert 1 <= len(groups.shape) <= 2
        assert len(groups) > 0 or facts.empty
        if facts.empty:
            self._peek = np.empty((len(min_times), 0), object)
            self._samples = np.empty((len(groups), len(min_times), 0), self.dtype)
            return
        self._peek = peek = self._analyze(facts, min_times, max_times, **kwargs)
        assert isinstance(peek, np.ndarray), type(self)
        assert peek.shape[:2] == (len(min_times), len(facts)), (peek.shape, type(self))
        if peek.dtype is np.dtype(object):
            notnull = peek != np.array(None)
        else:
            notnull = peek != peek.dtype.type(0)
        gpeek = [peek[:, g] for g in groups]
        gnotnull = [notnull[:, g] for g in groups]
        self._samples = [[p[nn].astype(self.dtype)
                          for p, nn in zip(gp, gnn)]
                         for gp, gnn in zip(gpeek, gnotnull)]

    @property
    def values(self) -> List[List[Metric[T]]]:
        """
        Return the current metric value.

        :return: groups x times -> metric.
        """
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
        """
        Calculate the samples for each item in the data frame.

        :return: np.ndarray[(len(times), len(facts))]
        """
        raise NotImplementedError

    def _value(self, samples: np.ndarray) -> Metric[T]:
        """Calculate the metric values from the current samples."""
        raise NotImplementedError

    def _values(self) -> List[List[Metric[T]]]:
        return [
            [self._value(self._cut_by_quantiles(s, self._calc_quantile_cut_values(
                np.concatenate(gs))))
             for s in gs]
            for gs in self._samples]

    def _calc_quantile_cut_values(self,
                                  samples: np.ndarray,
                                  as_dtype=None,
                                  ) -> Optional[np.ndarray]:
        """Calculate the quantile interval borders given the distribution and the desired \
        quantiles."""
        if (self._quantiles[0] == 0 and self._quantiles[1] == 1) or len(samples) == 0:
            return None
        if as_dtype is not None:
            samples = samples.astype(as_dtype)
        return np.quantile(samples, self._quantiles, interpolation="nearest")

    def _cut_by_quantiles(self,
                          samples: np.ndarray,
                          cut_values: Optional[np.ndarray],
                          ) -> np.ndarray:
        """Cut from the left and the right of the distribution by quantile cut values."""
        if len(samples) == 0 or cut_values is None:
            return samples
        if self._quantiles[0] != 0:
            samples = np.delete(samples, np.nonzero(samples < cut_values[0])[0])
        if self._quantiles[1] != 1:
            samples = np.delete(samples, np.nonzero(samples > cut_values[1])[0])
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
            negative = np.nonzero(samples < zero)[0]
            try:
                assert len(negative) == 0, samples[negative]
            except AssertionError as e:
                if sentry_sdk.Hub.current.scope.transaction is not None:
                    sentry_sdk.capture_exception(e)
                    samples = samples[samples >= zero]
                else:
                    raise e from None
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

    def _calc_quantile_cut_values(self,
                                  samples: np.ndarray,
                                  as_dtype=None,
                                  ) -> Optional[np.ndarray]:
        return super()._calc_quantile_cut_values(samples, as_dtype or self.deps[0].dtype)

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

    def _calc_quantile_cut_values(self,
                                  samples: np.ndarray,
                                  as_dtype=None,
                                  ) -> Optional[np.ndarray]:
        return None


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
            cut_values = self._calc_quantile_cut_values(np.concatenate(group_samples))
            histograms.append(group_histograms := [])
            for samples in group_samples:
                samples = self._cut_by_quantiles(samples, cut_values)
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
        """
        Calculate the current metric values.

        :return: Mapping from metric name to corresponding metric values (groups x times).
        """
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
def df_from_dataclasses(items: Iterable[Mapping[str, Any]],
                        length: Optional[int] = None,
                        ) -> pd.DataFrame:
    """
    Combine several dataclasses to a Pandas DataFrame.

    The dataclass type must be a Mapping. Pass `(i.__dict__ for i in items)` if it isn't.
    """
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
                assert isinstance(first_item, Mapping)
                for k in item:
                    columns[k] = []
            for k in item:
                columns[k].append(item[k])
    else:
        for i, item in enumerate(items):
            if i == 0:
                first_item = item
                assert isinstance(first_item, Mapping)
                for k in item:
                    columns[k] = [None] * length
            for k in item:
                columns[k][i] = item[k]
    if first_item is None:
        return pd.DataFrame()
    column_types = {}
    for k, v in type(first_item).__annotations__.items():
        if typing_utils.is_optional(v):
            if issubclass(unboxed := v.__args__[0], (datetime, np.datetime64, float)):
                # we can only unbox types that have a "NaN" value
                v = unboxed
            else:
                v = object
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


@sentry_span
def df_from_structs(items: Iterable[typing_utils.NumpyStruct],
                    length: Optional[int] = None,
                    ) -> pd.DataFrame:
    """
    Combine several NumpyStruct-s to a Pandas DataFrame.

    :param items: A collection, a generator, an iterator - all are accepted.
    :param length: In case `items` does not support `len()`, specify the number of structs \
                   for better performance.
    :return: Pandas DataFrame with columns set to struct fields.
    """
    columns = {}
    try:
        if length is None:
            length = len(items)
    except TypeError:
        # slower branch without pre-allocation
        items_iter = iter(items)
        try:
            first_item = next(items_iter)
        except StopIteration:
            return pd.DataFrame()
        assert isinstance(first_item, typing_utils.NumpyStruct)
        dtype = first_item.dtype
        nested_fields = first_item.nested_dtypes
        coerced_datas = [first_item.coerced_data]
        for k, v in first_item.items():
            if k not in dtype.names or k in nested_fields:
                columns[k] = [v]
        for item in items_iter:
            coerced_datas.append(item.coerced_data)
            for k in columns:
                columns[k].append(getattr(item, k))
        table_array = np.frombuffer(b"".join(coerced_datas), dtype=dtype)
        del coerced_datas
    else:
        items_iter = iter(items)
        try:
            first_item = next(items_iter)
        except StopIteration:
            return pd.DataFrame()
        assert isinstance(first_item, typing_utils.NumpyStruct)
        dtype = first_item.dtype
        nested_fields = first_item.nested_dtypes
        itemsize = dtype.itemsize
        coerced_datas = bytearray(itemsize * length)
        coerced_datas[:itemsize] = first_item.coerced_data
        for k, v in first_item.items():
            if k not in dtype.names or k in nested_fields:
                columns[k] = column = [None] * length
                column[0] = v
        for i, item in enumerate(items_iter, 1):
            coerced_datas[i * itemsize:(i + 1) * itemsize] = item.coerced_data
            for k in columns:
                columns[k][i] = item[k]
        table_array = np.frombuffer(coerced_datas, dtype=dtype)
        del coerced_datas
    for field_name in dtype.names:
        if field_name not in nested_fields:
            columns[field_name] = table_array[field_name]
    del table_array
    column_types = {}
    for k, v in first_item.optional.__annotations__.items():
        if not issubclass(v, (datetime, np.datetime64, float)):
            # we can only unbox types that have a "NaN" value
            v = object
        column_types[k] = v
    for k, v in columns.items():
        column_type = column_types.get(k, object)
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
                 metrics: Iterable[Sequence[str]],
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
        self._metrics = list(metrics)

    @sentry_span
    def __call__(self,
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
                 3D numpy array of List[List[Metric]]] (dtype object).
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
            """Return Array[List[List[M]]]."""
            cell = np.full(len(time_intervals), None, object)
            cell[:] = [[[] for _ in range(len(ts) - 1)] for ts in time_intervals]
            values_dict = values_dicts[ensemble_index]
            for metric in metrics[ensemble_index]:
                for tix, value in enumerate(values_dict[metric][group_index]):
                    primary, secondary = ts_index_map[tix]
                    cell[primary][secondary].append(value)
            return cell

        flat_vals = np.concatenate([
            fill_ensemble_group(m, g)
            for m in range(len(metrics))
            for g in range(flat_groups.shape[0])
        ])
        return flat_vals.reshape((len(metrics), *effective_groups_shape, len(time_intervals)))

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

    def _aggregate_ensembles(self, kwargs: Iterable[Mapping[str, Any]],
                             ) -> List[Dict[str, List[List[M]]]]:
        raise NotImplementedError


class BinnedMetricCalculator(BinnedEnsemblesCalculator[Metric]):
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
                 items: pd.DataFrame,
                 time_intervals: Sequence[Sequence[datetime]],
                 groups: np.ndarray,
                 ) -> np.ndarray:
        """
        Override the parent's method to reduce the level of nesting.

        :return: array of List[List[Metric]]].
        """
        return super().__call__(items, time_intervals, groups, [{}])[0]

    def _aggregate_ensembles(self, kwargs: Iterable[Mapping[str, Any]],
                             ) -> List[Dict[str, List[List[Metric]]]]:
        return [self.ensembles[0].values()]


class BinnedHistogramCalculator(BinnedEnsemblesCalculator[Histogram]):
    """Batched histograms calculation on sequential time intervals."""

    def _aggregate_ensembles(self, kwargs: Iterable[Mapping[str, Any]],
                             ) -> List[Dict[str, List[List[Histogram]]]]:
        return [{k: v for k, v in ensemble.histograms(**ekw).items()}
                for ensemble, ekw in zip(self.ensembles, kwargs)]


def group_to_indexes(items: pd.DataFrame,
                     *groupers: Callable[[pd.DataFrame], List[np.ndarray]],
                     ) -> np.ndarray:
    """Apply a chain of grouping functions to a table and return the tensor with group indexes."""
    if not groupers:
        return np.arange(len(items))[None, :]
    groups = [grouper(items) for grouper in groupers]

    def intersect(*coordinates: int) -> np.ndarray:
        return reduce(lambda x, y: np.intersect1d(x, y, assume_unique=True),
                      [group[i] for group, i in zip(groups, coordinates)])

    return np.fromfunction(np.vectorize(intersect, otypes=[object]),
                           [len(g) for g in groups], dtype=object)


def group_by_repo(repository_full_name_column_name: str,
                  repos: Sequence[Collection[str]],
                  items: pd.DataFrame,
                  ) -> List[np.ndarray]:
    """Group items by the value of their "repository_full_name" column."""
    if items.empty or items.empty:
        return [np.ndarray([], dtype=int)]
    repocol = items[repository_full_name_column_name].values.astype("U")
    order = np.argsort(repocol)
    unique_repos, pivots = np.unique(repocol[order], return_index=True)
    unique_indexes = np.split(np.arange(len(repocol))[order], pivots[1:])
    group_indexes = []
    for group in repos:
        indexes = []
        for repo in group:
            if unique_repos[(repo_index := searchsorted_inrange(unique_repos, repo)[0])] == repo:
                indexes.append(unique_indexes[repo_index])
        indexes = np.concatenate(indexes) if indexes else np.array([], dtype=int)
        group_indexes.append(indexes)
    return group_indexes


class RatioCalculator(WithoutQuantilesMixin, MetricCalculator[float]):
    """Calculate the ratio of two counts from the dependencies."""

    dtype = float

    def __init__(self, *deps: MetricCalculator, quantiles: Sequence[float]):
        """Initialize a new instance of RatioCalculator."""
        super().__init__(*deps, quantiles=quantiles)
        if isinstance(self._calcs[1], self.deps[0]):
            self._calcs = list(reversed(self._calcs))
        self._opened, self._closed = self._calcs

    def _values(self) -> List[List[Metric[float]]]:
        metrics = [[Metric(False, None, None, None)] * len(samples) for samples in self.samples]
        for i, (opened_group, closed_group) in enumerate(zip(
                self._opened.values, self._closed.values)):
            for j, (opened, closed) in enumerate(zip(opened_group, closed_group)):
                if not closed.exists and not opened.exists:
                    continue
                # Why +1? See ENG-866
                val = ((opened.value or 0) + 1) / ((closed.value or 0) + 1)
                metrics[i][j] = Metric(True, val, None, None)
        return metrics

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        return np.full((len(min_times), len(facts)), None, object)


def make_register_metric(metric_calculators: Dict[str, Type[MetricCalculator]],
                         histogram_calculators: Dict[str, Type[HistogramCalculator]]):
    """Create the decorator to keep track of the metric and histogram calculators."""
    def register_metric(name: str):
        assert isinstance(name, str)

        def register_with_name(cls: Type[MetricCalculator]):
            metric_calculators[name] = cls
            if not issubclass(cls, SumMetricCalculator):
                histogram_calculators[name] = \
                    type("HistogramOf" + cls.__name__, (cls, HistogramCalculator), {})
            return cls

        return register_with_name
    return register_metric
