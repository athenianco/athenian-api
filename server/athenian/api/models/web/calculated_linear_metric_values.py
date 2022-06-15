import datetime
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class CalculatedLinearMetricValues(Model):
    """Calculated metrics: date, values, confidences."""

    attribute_types = {
        "date": datetime.date,
        "values": List[object],
        "confidence_scores": Optional[List[int]],
        "confidence_mins": Optional[List[object]],
        "confidence_maxs": Optional[List[object]],
    }

    attribute_map = {
        "date": "date",
        "values": "values",
        "confidence_scores": "confidence_scores",
        "confidence_mins": "confidence_mins",
        "confidence_maxs": "confidence_maxs",
    }

    def __init__(
        self,
        date: Optional[datetime.date] = None,
        values: Optional[List[object]] = None,
        confidence_scores: Optional[List[int]] = None,
        confidence_mins: Optional[List[object]] = None,
        confidence_maxs: Optional[List[object]] = None,
    ):
        """Initialize CalculatedLinearMetricValues - a model defined in OpenAPI.

        :param date: The date of this CalculatedLinearMetricValues.
        :param values: The values of this CalculatedLinearMetricValues.
        :param confidence_mins: The left boundaries of the 80% confidence interval of this \
                               CalculatedLinearMetricValues.
        :param confidence_maxs: The right boundaries of the 80% confidence interval of this \
                               CalculatedLinearMetricValues.
        :param confidence_scores: The confidence scores of this CalculatedLinearMetricValues.
        """
        self._date = date
        self._values = values
        self._confidence_mins = confidence_mins
        self._confidence_maxs = confidence_maxs
        self._confidence_scores = confidence_scores

    @property
    def date(self) -> datetime.date:
        """Gets the date of this CalculatedLinearMetricValues.

        Where you should relate the metric value to on the time axis.

        :return: The date of this CalculatedLinearMetricValues.
        """
        return self._date

    @date.setter
    def date(self, date: datetime.date):
        """Sets the date of this CalculatedLinearMetricValues.

        Where you should relate the metric value to on the time axis.

        :param date: The date of this CalculatedLinearMetricValues.
        """
        if date is None:
            raise ValueError("Invalid value for `date`, must not be `None`")

        self._date = date

    @property
    def values(self) -> List[object]:
        """Gets the values of this CalculatedLinearMetricValues.

        The same order as `metrics`.

        :return: The values of this CalculatedLinearMetricValues.
        """
        return self._values

    @values.setter
    def values(self, values: List[object]):
        """Sets the values of this CalculatedLinearMetricValues.

        The same order as `metrics`.

        :param values: The values of this CalculatedLinearMetricValues.
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values

    @property
    def confidence_mins(self) -> Optional[List[object]]:
        """Gets the left boundaries of the 80% confidence interval of this \
        CalculatedLinearMetricValues.

        Confidence interval @ p=0.8, minimum. The same order as `metrics`.

        :return: The left boundaries of the 80% confidence interval of this \
                 CalculatedLinearMetricValues.
        """
        return self._confidence_mins

    @confidence_mins.setter
    def confidence_mins(self, confidence_mins: Optional[List[object]]):
        """Sets the left boundaries of the 80% confidence interval of this \
        CalculatedLinearMetricValues.

        Confidence interval @ p=0.8, minimum. he same order as `metrics`.

        :param confidence_mins: The left boundaries of the 80% confidence interval of this \
                                CalculatedLinearMetricValues.
        """
        self._confidence_mins = confidence_mins

    @property
    def confidence_maxs(self) -> Optional[List[object]]:
        """Gets the right boundaries of the 80% confidence interval of this \
        CalculatedLinearMetricValues.

        Confidence interval @ p=0.8, maximum. The same order as `metrics`.

        :return: The right boundaries of the 80% confidence interval of this \
                 CalculatedLinearMetricValues.
        """
        return self._confidence_maxs

    @confidence_maxs.setter
    def confidence_maxs(self, confidence_maxs: Optional[List[object]]):
        """Sets the right boundaries of the 80% confidence interval of this \
        CalculatedLinearMetricValues.

        Confidence interval @ p=0.8, maximum. he same order as `metrics`.

        :param confidence_maxs: The right boundaries of the 80% confidence interval of this \
                                CalculatedLinearMetricValues.
        """
        self._confidence_maxs = confidence_maxs

    @property
    def confidence_scores(self) -> Optional[List[int]]:
        """Gets the confidence scores of this CalculatedLinearMetricValues.

        The same order as `metrics`.

        :return: The values of this CalculatedLinearMetricValues.
        """
        return self._confidence_scores

    @confidence_scores.setter
    def confidence_scores(self, confidence_scores: Optional[List[int]]):
        """Sets the confidence scores of this CalculatedLinearMetricValues.

        Confidence score from 0 (no idea) to 100 (very confident). The same order as `metrics`.

        :param confidence_scores: The confidence scores of this CalculatedLinearMetricValues.
        """
        self._confidence_scores = confidence_scores
