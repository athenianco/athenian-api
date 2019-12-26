from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

T = TypeVar("T", float, int, datetime)


@dataclass(frozen=True)
class Metric(Generic[T]):
    """Any statistic measurement bundled with the corresponding confidence interval."""

    exists: bool
    value: T
    confidence_min: T
    confidence_max: T
