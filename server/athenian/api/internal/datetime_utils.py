from datetime import date, datetime, time, timedelta, timezone
from typing import List, Optional, Tuple, Union

from athenian.api.models.web import Granularity, InvalidRequestError
from athenian.api.response import ResponseError


def coarsen_time_interval(time_from: datetime, time_to: datetime) -> Tuple[date, date]:
    """Extend the time interval to align at the date boarders."""
    assert time_to > time_from
    zerotd = timedelta(0)
    for t in time_to, time_from:
        assert (
            isinstance(t, datetime)
            and t.tzinfo is not None
            and t.tzinfo.utcoffset(time_from) == zerotd
        )
    date_from = time_from.date()
    date_to = time_to.date()
    if time_to.time() != datetime.min.time():
        date_to += timedelta(days=1)
    return date_from, date_to


def split_to_time_intervals(
    date_from: date,
    date_to: date,
    granularities: str | list[str],
    tzoffset: Optional[int],
) -> Tuple[Union[List[datetime], List[List[datetime]]], timedelta]:
    """Produce time interval boundaries from the min and the max dates and the interval lengths \
    (granularities).

    :param tzoffset: Time zone offset in minutes. We ignore DST for now.
    :return: tuple with the time intervals and the timezone offset converted to timedelta. \
             If `granularities` is a scalar, then return a list of boundaries, otherwise, return \
             a list of lists.
    """
    if date_to < date_from:
        raise ResponseError(
            InvalidRequestError(
                detail="date_from may not be greater than date_to",
                pointer=".date_from",
            ),
        )
    tz_timedelta = timedelta(minutes=-tzoffset) if tzoffset is not None else timedelta(0)

    def split(granularity: str, ptr: str) -> List[datetime]:
        try:
            intervals = Granularity.split(granularity, date_from, date_to)
        except ValueError:
            raise ResponseError(
                InvalidRequestError(
                    detail='granularity "%s" does not match /%s/'
                    % (granularity, Granularity.format.pattern),
                    pointer=ptr,
                ),
            )
        return [
            datetime.combine(i, datetime.min.time(), tzinfo=timezone.utc) + tz_timedelta
            for i in intervals
        ]

    if isinstance(granularities, str):
        return split(granularities, ".granularity"), tz_timedelta

    return [split(g, ".granularities[%d]" % i) for i, g in enumerate(granularities)], tz_timedelta


def closed_dates_interval_to_datetimes(from_: date, to_: date) -> tuple[datetime, datetime]:
    """Convert a closed dates interval to a datetimes interval.

    Dates interval is closed in the sense that both `from_` and `to_` days are included
    in the interval.

    """
    from_dt = datetime.combine(from_, time.min, tzinfo=timezone.utc)
    # set datetime to the start of the following day
    to_dt = datetime.combine(to_ + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return (from_dt, to_dt)


def datetimes_to_closed_dates_interval(from_: datetime, to_: datetime) -> tuple[date, date]:
    """Convert a datetime interval to a closed dates interval.

    This is the inverse function of `closed_dates_interval_to_datetimes`.
    """
    return (from_.date(), to_.date() - timedelta(days=1))
