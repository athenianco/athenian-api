from datetime import datetime, timezone
from typing import Any

import numpy as np


def dt(*args: Any) -> datetime:
    """Shortcut to generate a UTC date."""

    return datetime(*args, tzinfo=timezone.utc)


def dt64arr_us(dt: datetime) -> np.ndarray:
    return np.array([dt.replace(tzinfo=None)], dtype="datetime64[us]")


def dt64arr_s(dt: datetime) -> np.ndarray:
    return np.array([dt.replace(tzinfo=None)], dtype="datetime64[s]")
