from datetime import date, datetime, time, timedelta, timezone
from typing import Tuple


def goal_dates_to_datetimes(valid_from: date, expires_at: date) -> Tuple[datetime, datetime]:
    """Convert goal dates from API into datetimes."""
    valid_from = datetime.combine(valid_from, time.min, tzinfo=timezone.utc)
    # expiresAt semantic is to include the given day, so datetime is set to the start of the
    # following day
    expires_at = datetime.combine(expires_at + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return (valid_from, expires_at)


def goal_datetimes_to_dates(valid_from: datetime, expires_at: datetime) -> Tuple[date, date]:
    """Convert datetime objects into API dates.

    This is the inverse function of `goal_dates_to_datetimes`.
    """
    return (valid_from.date(), expires_at.date() - timedelta(days=1))
