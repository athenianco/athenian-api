from datetime import datetime, timezone
from typing import Any


def dt(*args: Any) -> datetime:
    """Shortcut to generate a UTC date."""

    return datetime(*args, tzinfo=timezone.utc)
