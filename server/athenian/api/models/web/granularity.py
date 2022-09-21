from datetime import date, datetime, timedelta
import re
from typing import List

from dateutil.rrule import DAILY, MONTHLY, WEEKLY, YEARLY, rrule

from athenian.api.models.web.base_model_ import Model


class Granularity(Model):
    """Time frequency."""

    format = re.compile(r"all|(([1-9]\d* )?(aligned )?(day|week|month|year))")

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
        match = cls.format.fullmatch(value)
        if not match:
            raise ValueError("Invalid granularity format: " + value)
        if value == "all":
            return [date_from, date_to + timedelta(days=1)]
        _, step, aligned, base = match.groups()
        if step is None:
            step = 1
        freq = {
            "day": DAILY,
            "week": WEEKLY,
            "month": MONTHLY,
            "year": YEARLY,
        }[base]
        if aligned and base != "day":
            alignment = {"by%sday" % base: 1 if base != "week" else 0}
        else:
            alignment = {}
        unsampled = [d.date() for d in rrule(freq, dtstart=date_from, until=date_to, **alignment)]
        if not unsampled or unsampled[0] != date_from:
            # happens because of the alignment
            unsampled.insert(0, date_from)
        series = unsampled[:: int(step)]
        series.append(date_to + timedelta(days=1))
        return series


class GranularityMixin:
    """Implement `granularity` property."""

    def validate_granularity(self, granularity: str) -> str:
        """Sets the granularity of this model.

        :param granularity: The granularity of this model.
        """
        if granularity is None:
            raise ValueError("Invalid value for `granularity`, must not be `None`")
        if not Granularity.format.match(granularity):
            raise ValueError(
                'Invalid value for `granularity`: "%s" does not match /%s/' % granularity,
                Granularity.format.pattern,
            )

        return granularity
