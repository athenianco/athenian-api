from datetime import date, datetime, time, timedelta, timezone
from typing import Tuple

from dateutil.relativedelta import relativedelta


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


def goal_initial_query_interval(
    valid_from: datetime,
    expires_at: datetime,
) -> Tuple[datetime, datetime]:
    """Return the time interval to query the initial metric value for the goal.

    `valid_from` and `expires_at` parameter are the `datetime`-s stored on DB for the Goal.

    """
    # unlike timedelta, dateutil relativedelta understands months, so the interval will
    # be better aligned
    current_span = relativedelta(expires_at, valid_from)

    initial_at = valid_from
    initial_from = valid_from - current_span

    return (initial_from, initial_at)
