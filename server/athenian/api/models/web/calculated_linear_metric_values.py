import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class CalculatedLinearMetricValues(Model):
    """Calculated metrics: date, values, confidences."""

    date: datetime.date
    values: list[object]
    confidence_scores: Optional[list[int]]
    confidence_mins: Optional[list[object]]
    confidence_maxs: Optional[list[object]]
