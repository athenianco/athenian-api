from datetime import date
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class CalculatedLinearMetricValues(Model):
    """Calculated metrics: date, values, confidences."""

    date: date
    values: list[object]
    confidence_scores: Optional[list[int]]
    confidence_mins: Optional[list[object]]
    confidence_maxs: Optional[list[object]]

    def clean_up_confidence_fields(self) -> None:
        """Clear the confidence fields when `confidence_scores` contains nothing, in place."""
        if all(score is None for score in self.confidence_scores or ()):
            self.confidence_mins = None
            self.confidence_maxs = None
            self.confidence_scores = None
