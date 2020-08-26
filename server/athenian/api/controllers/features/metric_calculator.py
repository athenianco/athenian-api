from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, Type

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

    def __init__(self, *deps: "MetricCalculator", quantiles: Sequence[float]):
        """Initialize a new `MetricCalculator` instance."""
        self._samples = np.array([], dtype=object)
        self._peek = np.array([], dtype=object)
        self._last_value = None
        self._calcs = []
        self._quantiles = np.asarray(quantiles)
        for calc in deps:
            for cls in self.deps:
                if isinstance(calc, cls):
                    self._calcs.append(calc)

    def __call__(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> None:
        """Supply another pull request facts to update the state.

        :param facts: Mined facts about another pull request: timestamps, stats, etc.
        :param min_time: Start of the considered time interval. It is needed to discard samples \
                         with both ends less than the minimum time.
        :param max_time: Finish of the considered time interval. It is needed to discard samples \
                         with both ends greater than the maximum time.
        """
        assert isinstance(facts, pd.DataFrame)
        assert min_time.tzinfo is None
        assert max_time.tzinfo is None
        if facts.empty:
            self._peek = np.array([], dtype=object)
            self._samples = np.array([], dtype=object)
            return
        peek = self._peek = self._analyze(facts, min_time, max_time, **kwargs)
        samples = self._samples = peek[peek != np.array(None)].astype(self.dtype)
        assert isinstance(samples, np.ndarray), self

    def reset(self):
        """Reset the internal state."""
        self._peek = np.array([], dtype=object)
        self._samples = np.array([], dtype=object)
        self._last_value = None
        for calc in self._calcs:
            calc.reset()

    @property
    def value(self) -> Metric[T]:
        """Return the current metric value."""
        if self._last_value is None:
            self._last_value = self._value(self._cut_by_quantiles())
        return self._last_value

    @property
    def peek(self) -> np.ndarray:
        """Return the last calculated samples, with None-s."""
        return self._peek

    @property
    def samples(self) -> np.ndarray:
        """Return the last calculated samples, without None-s."""
        return self._samples

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        """Calculate the samples for each item in the data frame."""
        raise NotImplementedError

    def _value(self, samples: np.ndarray) -> Metric[T]:
        """Calculate the metric value from the specified samples."""
        raise NotImplementedError

    def _cut_by_quantiles(self) -> np.ndarray:
        """Cut from the left and the right of the distribution by quantiles."""
        samples = self.samples
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
        return Metric(exists, samples.sum() if exists else None, None, None)

    def _analyze(self, facts: Any, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[T]:
        raise NotImplementedError


class Counter(MetricCalculator[int]):
    """Count the number of PRs that were used to calculate the specified metric."""

    dtype = int

    def _value(self, samples: np.ndarray) -> Metric[int]:
        return Metric(True, len(samples), None, None)

    def _analyze(self, facts: Any, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        return self._calcs[0].peek


class WithoutQuantilesMixin:
    """Ignore the quantiles."""

    def _cut_by_quantiles(self) -> np.ndarray:
        return self._samples


class HistogramCalculator(MetricCalculator):
    """Pull request histogram calculator, base abstract class."""

    def histogram(self, scale: Scale, bins: int) -> Histogram[T]:
        """Calculate the histogram over the current distribution."""
        samples = self._cut_by_quantiles()
        if scale == Scale.LOG:
            shift_log = getattr(self, "_shift_log", None)  # type: Optional[Callable[[T], T]]
            if shift_log is not None:
                samples = np.array([shift_log(s) for s in self.samples])
        return calculate_histogram(samples, scale, bins)


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

    def __call__(self, facts: Any, min_time: datetime, max_time: datetime,
                 **kwargs) -> None:
        """Consume another pull request by the owned calculators, changing their states."""
        for calc in self._calcs:
            calc(facts, min_time, max_time, **kwargs)

    def __bool__(self) -> bool:
        """Return True if there is at leat one calculator inside; otherwise, False."""
        return bool(len(self._calcs))

    def values(self) -> Dict[str, Metric[T]]:
        """Calculate the current metric values."""
        return {k: v.value for k, v in self._metrics.items()}

    def reset(self) -> None:
        """Reset the internal states of all the owned calculators."""
        for calc in self._calcs:
            calc.reset()


class HistogramCalculatorEnsemble(MetricCalculatorEnsemble):
    """Like MetricCalculatorEnsemble, but for histograms."""

    def histograms(self, scale: Scale, bins: int) -> Dict[str, Histogram]:
        """Calculate the current histograms."""
        return {k: v.histogram(scale, bins) for k, v in self._metrics.items()}


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


class BinnedMetricCalculator:
    """Batched metrics calculation on sequential time intervals."""

    def __init__(self,
                 metrics: Sequence[str],
                 time_intervals: Sequence[datetime],
                 quantiles: Sequence[float],
                 class_mapping: Dict[str, Type[MetricCalculator]],
                 start_time_getter: Callable[[Any], datetime],
                 finish_time_getter: Callable[[Any], Optional[datetime]]):
        """
        Initialize a new instance of `BinnedMetricCalculator`.

        :param metrics: Sequence of metric names to calculate in each bin.
        :param time_intervals: Time interval borders in UTC. Each interval spans \
                               `[time_intervals[i], time_intervals[i + 1]]`, the ending \
                               not included.
        :param class_mapping: Mapping from metric names to the corresponding calculator classes.
        :param start_time_getter: Extractor of the time when the object appears. \
                                  It is used for the sorting optimization.
        :param finish_time_getter: Extractor of the time when the object disappears. \
                                   It is used for the sorting optimization. \
                                   Note: the object may be still active, hence an Optional return.
        """
        self.calcs = MetricCalculatorEnsemble(
            *metrics, quantiles=quantiles, class_mapping=class_mapping)
        self.metrics = metrics
        assert len(time_intervals) >= 2
        self.time_intervals = time_intervals
        self.start_time_getter = start_time_getter
        self.finish_time_getter = finish_time_getter

    def __call__(self, items: Iterable[Any]) -> List[List[Metric]]:
        """Calculate the binned metrics on a series of objects."""
        items = df_from_dataclasses(items)
        result = []
        for time_from, time_to in zip(self.time_intervals, self.time_intervals[1:]):
            assert time_from.tzinfo is not None
            assert time_to.tzinfo is not None
            self.calcs(items, time_from.replace(tzinfo=None), time_to.replace(tzinfo=None))
            values_dict = self.calcs.values()
            result.append([values_dict[m] for m in self.metrics])
            self.calcs.reset()
        return result
