from dataclasses import dataclass
from datetime import timedelta
from types import new_class
from typing import Any, Generic, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np

from athenian.api.typing_utils import numpy_struct

T = TypeVar("T", float, int, timedelta, None)
np.seterr(divide="raise")


class Metric(Generic[T]):
    """Any statistic measurement bundled with the corresponding confidence interval."""

    @property
    def exists(self) -> bool:
        """Return whether the metric exists."""
        try:
            return super().exists
        except AttributeError:
            raise NotImplementedError from None

    @property
    def value(self) -> Optional[T]:
        """Return the metric's value."""
        try:
            return super().value
        except AttributeError:
            raise NotImplementedError from None

    @property
    def confidence_min(self) -> Optional[T]:
        """Return the start of the confidence interval."""
        try:
            return super().confidence_min
        except AttributeError:
            raise NotImplementedError from None

    @property
    def confidence_max(self) -> Optional[T]:
        """Return the finish of the confidence interval."""
        try:
            return super().confidence_max
        except AttributeError:
            raise NotImplementedError from None

    def confidence_score(self) -> Optional[int]:
        """Calculate the confidence score from 0 (garbage) to 100 (very confident)."""
        raise NotImplementedError


class NumpyMetric(Metric[T]):
    """Base class for all @numpy_struct metrics."""

    nan = None  # numpy-compatible "missing" value

    @classmethod
    def from_fields(
        cls,
        exists: bool,
        value: Optional[T],
        confidence_min: Optional[T],
        confidence_max: Optional[T],
    ) -> "NumpyMetric":
        """Initialize a new instance of NumpyStruct from the mapping of immutable field \
        values."""
        if not exists:
            assert value is None
            assert confidence_min is None
            assert confidence_max is None
            value = confidence_min = confidence_max = cls.nan
        else:
            assert value is not None
            if confidence_min is None:
                confidence_min = cls.nan
            if confidence_max is None:
                confidence_max = cls.nan
        return super().from_fields(
            exists=exists,
            value=value,
            confidence_min=confidence_min,
            confidence_max=confidence_max,
        )

    def __repr__(self) -> str:
        """Replace repr() of NumpyStruct."""
        return (
            f"{type(self).__name__}.from_fields({self.exists}, "
            f"{self.value}, {self.confidence_min}, {self.confidence_max})"
        )

    @property
    def exists(self) -> bool:
        """Return whether the metric exists."""
        return super().exists

    @property
    def value(self) -> Optional[T]:
        """Return the metric's value."""
        v = super().value
        if v is not None and v != self.nan:
            return v.item()
        return None

    @property
    def confidence_min(self) -> Optional[T]:
        """Return the start of the confidence interval."""
        v = super().confidence_min
        if v is not None and v != self.nan:
            return v.item()
        return None

    @property
    def confidence_max(self) -> Optional[T]:
        """Return the finish of the confidence interval."""
        v = super().confidence_max
        if v is not None and v != self.nan:
            return v.item()
        return None

    def confidence_score(self) -> Optional[int]:
        """Calculate the confidence score from 0 (garbage) to 100 (very confident)."""
        if self.confidence_min is None and self.confidence_max is None:
            return None
        try:
            eps = min(100 * ((self.confidence_max - self.confidence_min) / self.value), 100)
            return 100 - int(eps)
        except (ZeroDivisionError, FloatingPointError):
            if self.confidence_min == self.confidence_max == self.value:
                return 100  # everything is zero so no worries
            return 0  # we really don't know the score in this case


def make_metric(
    name: str,
    module: str,
    dtype: Union[str, T, np.dtype],
    nan_value: Any,
) -> Type[NumpyMetric]:
    """Generate a new NumpyMetric specialization."""
    vulgar_type = type(np.zeros(1, dtype=np.dtype(dtype)).item())

    class Immutable:
        exists: bool
        value: dtype
        confidence_min: dtype
        confidence_max: dtype

    # we cannot use the normal "class" construct because the class name must be `name`, and that
    # must be because we need to pickle
    cls = numpy_struct(
        new_class(
            name,
            (NumpyMetric[vulgar_type],),
            {},
            lambda ns: ns.update(
                nan=nan_value,
                Immutable=Immutable,
            ),
        ),
    )
    cls.__doc__ = f"NumpyMetric specialization for {dtype}."
    cls.__module__ = module
    return cls


MetricTimeDelta = make_metric("MetricTimeDelta", __name__, "timedelta64[s]", np.timedelta64("NaT"))
MetricInt = make_metric("MetricInt", __name__, int, np.iinfo(int).min)
MetricFloat = make_metric("MetricFloat", __name__, np.float32, np.nan)


@dataclass(slots=True, frozen=True, init=False)
class MultiMetric(Metric[T]):
    """Several metrics of the same kind grouped together."""

    metrics: Tuple[Metric[T]]

    def __init__(self, *metrics: Metric[T]):
        """Wrap several Metric-s together."""
        object.__setattr__(self, "metrics", metrics)

    @property
    def exists(self) -> bool:
        """Return True if at least one internal metric exists."""
        return any(m.exists for m in self.metrics)

    @property
    def value(self) -> List[T]:
        """Return values of the internal metrics."""
        return [m.value for m in self.metrics]

    @property
    def confidence_min(self) -> List[T]:
        """Return confidence mins of the internal metrics."""
        return [m.confidence_min for m in self.metrics]

    @property
    def confidence_max(self) -> List[T]:
        """Return confidence maxs of the internal metrics."""
        return [m.confidence_max for m in self.metrics]

    def confidence_score(self) -> Optional[List[int]]:
        """Return confidence scores of the internal metrics."""
        return [m.confidence_score() for m in self.metrics]
