from datetime import timedelta
from typing import Generic, Optional, TypeVar

import numpy as np

from athenian.api.typing_utils import dataclass


T = TypeVar("T", float, int, timedelta, type(None))
np.seterr(divide="raise")


@dataclass(slots=True, frozen=True, init=False)
class Metric(Generic[T]):
    """Any statistic measurement bundled with the corresponding confidence interval."""

    exists: bool
    value: Optional[T]
    confidence_min: Optional[T]
    confidence_max: Optional[T]

    def __init__(self,
                 exists: bool,
                 value: Optional[T],
                 confidence_min: Optional[T],
                 confidence_max: Optional[T]):
        """Initialize a new instance of Metric."""
        object.__setattr__(self, "exists", exists)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "confidence_min", confidence_min)
        object.__setattr__(self, "confidence_max", confidence_max)
        if not exists:
            assert value is None
            assert confidence_min is None
            assert confidence_max is None
        else:
            assert value is not None

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
