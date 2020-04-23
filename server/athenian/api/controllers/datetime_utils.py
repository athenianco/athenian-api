from datetime import date, datetime, timedelta
from typing import Tuple


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
