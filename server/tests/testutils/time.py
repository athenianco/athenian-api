from datetime import datetime, timezone
from typing import Any

import freezegun
import numpy as np


def dt(*args: Any) -> datetime:
    """Shortcut to generate a UTC date."""

    return datetime(*args, tzinfo=timezone.utc)


def dt64arr_ns(dt: datetime) -> np.ndarray:
    return np.array([dt.replace(tzinfo=None)], dtype="datetime64[ns]")


def dt64arr_s(dt: datetime) -> np.ndarray:
    return np.array([dt.replace(tzinfo=None)], dtype="datetime64[s]")


def freeze_time(*args: Any, **kwargs: Any) -> freezegun.api._freeze_time:
    """Wrapper over freezefun.freeze_time to avoid patching problematic modules."""
    ignore = [
        # needed to not break klass == date tests
        "athenian.api.serialization",
    ]
    kwargs["ignore"] = [*kwargs.get("ignore", ()), *ignore]
    res = freezegun.freeze_time(*args, **kwargs)
    return res
