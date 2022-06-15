from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple, Union

from athenian.api.models.web import Granularity, InvalidRequestError
from athenian.api.response import ResponseError


def coarsen_time_interval(time_from: datetime, time_to: datetime) -> Tuple[date, date]:
    """Extend the time interval to align at the date boarders."""
    assert time_to > time_from
    zerotd = timedelta(0)
    assert isinstance(time_from, datetime) and time_from.tzinfo.utcoffset(time_from) == zerotd
    assert isinstance(time_to, datetime) and time_to.tzinfo.utcoffset(time_to) == zerotd
    date_from = time_from.date()
    date_to = time_to.date()
    if time_to.time() != datetime.min.time():
        date_to += timedelta(days=1)
    return date_from, date_to


def split_to_time_intervals(
    date_from: date,
    date_to: date,
    granularities: Union[str, List[str]],
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
    tzoffset = timedelta(minutes=-tzoffset) if tzoffset is not None else timedelta(0)

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
            datetime.combine(i, datetime.min.time(), tzinfo=timezone.utc) + tzoffset
            for i in intervals
        ]

    if isinstance(granularities, str):
        return split(granularities, ".granularity"), tzoffset

    return [split(g, ".granularities[%d]" % i) for i, g in enumerate(granularities)], tzoffset
