from datetime import date
import re
from typing import List

from dateutil.rrule import DAILY, MONTHLY, rrule, WEEKLY, YEARLY

from athenian.api.models.web.base_model_ import Model


class Granularity(Model):
    """Time frequency."""

    format = re.compile(r"^([1-9]\d* )?(day|week|month|year)$")

    def __init__(self):
        """Granularity - a model defined in OpenAPI"""
        self.openapi_types = {}

        self.attribute_map = {}

    @classmethod
    def split(cls, value: str, date_from: date, date_to: date) -> List[date]:
        """
        Cut [date_from -> date_to] into evenly sized time intervals.

        Special handling of "month" is required so that each interval spans over one month.
        """
        match = cls.format.match(value)
        if not match:
            raise ValueError("Invalid granularity format: " + value)
        step, base = match.groups()
        if step is None:
            step = 1
        freq = {
            "day": DAILY,
            "week": WEEKLY,
            "month": MONTHLY,
            "year": YEARLY,
        }[base]
        unsampled = [d.date() for d in rrule(freq, dtstart=date_from, until=date_to)]
        return unsampled[::int(step)]
