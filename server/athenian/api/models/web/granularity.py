from datetime import date, datetime
import re
from typing import List

from dateutil.rrule import DAILY, MONTHLY, rrule, WEEKLY, YEARLY

from athenian.api.models.web.base_model_ import Model


class Granularity(Model):
    """Time frequency."""

    format = re.compile(r"^(([1-9]\d* )?(day|week|month|year)|all)$")

    @classmethod
    def split(cls, value: str, date_from: date, date_to: date) -> List[date]:
        """
        Cut [date_from -> date_to] into evenly sized time intervals.

        Special handling of "month" is required so that each interval spans over one month.
        The last element of the returned series always equals `date_to`. That is, the last
        time interval may be shorter than requested.
        """
        assert date_from <= date_to
        assert isinstance(date_from, date) and not isinstance(date_from, datetime)
        assert isinstance(date_to, date) and not isinstance(date_to, datetime)
        match = cls.format.match(value)
        if not match:
            raise ValueError("Invalid granularity format: " + value)
        if value == "all":
            return [date_from, date_to]
        _, step, base = match.groups()
        if step is None:
            step = 1
        freq = {
            "day": DAILY,
            "week": WEEKLY,
            "month": MONTHLY,
            "year": YEARLY,
        }[base]
        unsampled = [d.date() for d in rrule(freq, dtstart=date_from, until=date_to)]
        series = unsampled[::int(step)]
        if series[-1] != date_to or len(series) == 1:
            series.append(date_to)
        return series
