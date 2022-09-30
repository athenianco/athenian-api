from __future__ import annotations

import dataclasses
from datetime import date, datetime, time, timedelta, timezone

from dateutil.relativedelta import relativedelta

from athenian.api.models.web.goal import GoalSeriesGranularity
from athenian.api.models.web.granularity import Granularity


def goal_dates_to_datetimes(valid_from: date, expires_at: date) -> tuple[datetime, datetime]:
    """Convert goal dates from API into datetimes."""
    valid_from = datetime.combine(valid_from, time.min, tzinfo=timezone.utc)
    # expiresAt semantic is to include the given day, so datetime is set to the start of the
    # following day
    expires_at = datetime.combine(expires_at + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return (valid_from, expires_at)


def goal_datetimes_to_dates(valid_from: datetime, expires_at: datetime) -> tuple[date, date]:
    """Convert datetime objects into API dates.

    This is the inverse function of `goal_dates_to_datetimes`.
    """
    return (valid_from.date(), expires_at.date() - timedelta(days=1))


def goal_initial_query_interval(
    valid_from: datetime,
    expires_at: datetime,
) -> tuple[datetime, datetime]:
    """Return the time interval to query the initial metric value for the goal.

    `valid_from` and `expires_at` parameter are the `datetime`-s stored on DB for the Goal.
    """
    # unlike timedelta, dateutil relativedelta understands months, so the interval will
    # be better aligned
    current_span = relativedelta(expires_at, valid_from)

    initial_at = valid_from
    initial_from = valid_from - current_span

    return (initial_from, initial_at)


Intervals = tuple[datetime, ...]


@dataclasses.dataclass(slots=True, frozen=True)
class GoalTimeseriesSpec:
    """The specification for a timeseries collecting goal metric values."""

    intervals: Intervals
    granularity: GoalSeriesGranularity

    @classmethod
    def from_timespan(cls, valid_from: datetime, expires_at: datetime) -> GoalTimeseriesSpec:
        """Build the intervals given the timespan of the goal, as saved on DB."""
        granularity = cls._granularity(valid_from, expires_at)
        # Granularity.split accepts an interval where the last day is included
        from_date, at_date = goal_datetimes_to_dates(valid_from, expires_at)

        date_intervals = tuple(Granularity.split(granularity.value, from_date, at_date))
        dt_intervals = tuple(
            datetime.combine(d, time.min, tzinfo=timezone.utc) for d in date_intervals
        )
        return GoalTimeseriesSpec(dt_intervals, granularity.value)

    @classmethod
    def _granularity(cls, valid_from: datetime, expires_at: datetime) -> GoalSeriesGranularity:
        if (expires_at - valid_from) <= timedelta(days=93):  # goal with a quarter timespan or less
            return GoalSeriesGranularity.WEEK
        return GoalSeriesGranularity.MONTH
