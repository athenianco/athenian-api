# coding: utf-8

from datetime import date
from typing import List, Optional

from athenian.api import util
from athenian.api.models.base_model_ import Model


class CalculatedMetricValues(Model):
    """Calculated metrics: date, values, confidences."""

    def __init__(self, date: date, values: List[float],
                 confidence_scores: List[int],
                 confidence_mins: Optional[List[float]] = None,
                 confidence_maxs: Optional[List[float]] = None,
                 ):
        """Initialize CalculatedMetricValues - a model defined in OpenAPI.

        :param date: The date of this CalculatedMetricValues.
        :param values: The values of this CalculatedMetricValues.
        :param confidence_mins: The left boundaries of the 95% confidence interval of this \
                               CalculatedMetricValues.
        :param confidence_maxs: The right boundaries of the 95% confidence interval of this \
                               CalculatedMetricValues.
        :param confidence_scores: The confidence scores of this CalculatedMetricValues.
        """
        self.openapi_types = {"date": date, "values": List[float],
                              "confidence_scores": List[int],
                              "confidence_mins": List[float],
                              "confidence_maxs": List[float],
                              }
        self.attribute_map = {"date": "date", "values": "values",
                              "confidence_scores": "confidence_scores",
                              "confidence_mins": "confidence_mins",
                              "confidence_maxs": "confidence_maxs",
                              }
        self.date = date
        self.values = values
        self.confidence_mins = confidence_mins
        self.confidence_maxs = confidence_maxs
        self.confidence_scores = confidence_scores

    @classmethod
    def from_dict(cls, dikt: dict) -> "CalculatedMetricValues":
        """Returns the dict as a model.

        :param dikt: A dict.
        :return: The CalculatedMetric_values of this CalculatedMetricValues.
        """
        return util.deserialize_model(dikt, cls)

    @property
    def date(self) -> date:
        """Gets the date of this CalculatedMetricValues.

        Where you should relate the metric value to on the time axis.

        :return: The date of this CalculatedMetricValues.
        :rtype: date
        """
        return self._date

    @date.setter
    def date(self, date: date):
        """Sets the date of this CalculatedMetricValues.

        Where you should relate the metric value to on the time axis.

        :param date: The date of this CalculatedMetricValues.
        :type date: date
        """
        if date is None:
            raise ValueError("Invalid value for `date`, must not be `None`")

        self._date = date

    @property
    def values(self) -> List[float]:
        """Gets the values of this CalculatedMetricValues.

        The same order as `metrics`.

        :return: The values of this CalculatedMetricValues.
        :rtype: List[float]
        """
        return self._values

    @values.setter
    def values(self, values: List[float]):
        """Sets the values of this CalculatedMetricValues.

        The same order as `metrics`.

        :param values: The values of this CalculatedMetricValues.
        :type values: List[float]
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values

    @property
    def confidence_mins(self) -> List[float]:
        """Gets the left boundaries of the 95% confidence interval of this CalculatedMetricValues.

        Confidence interval @ p=0.95, minimum. The same order as `metrics`.

        :return: The left boundaries of the 95% confidence interval of this CalculatedMetricValues.
        :rtype: List[float]
        """
        return self._confidence_mins

    @confidence_mins.setter
    def confidence_mins(self, confidence_mins: List[float]):
        """Sets the left boundaries of the 95% confidence interval of this CalculatedMetricValues.

        Confidence interval @ p=0.95, minimum. he same order as `metrics`.

        :param confidence_mins: The left boundaries of the 95% confidence interval of this \
                               CalculatedMetricValues.
        :type confidence_mins: List[float]
        """
        self._confidence_mins = confidence_mins

    @property
    def confidence_maxs(self) -> List[float]:
        """Gets the right boundaries of the 95% confidence interval of this CalculatedMetricValues.

        Confidence interval @ p=0.95, maximum. The same order as `metrics`.

        :return: The right boundaries of the 95% confidence interval of this \
                 CalculatedMetricValues.
        :rtype: List[float]
        """
        return self._confidence_maxs

    @confidence_maxs.setter
    def confidence_maxs(self, confidence_maxs: List[float]):
        """Sets the right boundaries of the 95% confidence interval of this CalculatedMetricValues.

        Confidence interval @ p=0.95, maximum. he same order as `metrics`.

        :param confidence_maxs: The right boundaries of the 95% confidence interval of this \
                               CalculatedMetricValues.
        :type confidence_maxs: List[float]
        """
        self._confidence_maxs = confidence_maxs

    @property
    def confidence_scores(self) -> List[int]:
        """Gets the confidence scores of this CalculatedMetricValues.

        The same order as `metrics`.

        :return: The values of this CalculatedMetricValues.
        :rtype: List[int]
        """
        return self._confidence_scores

    @confidence_scores.setter
    def confidence_scores(self, confidence_scores: List[int]):
        """Sets the confidence scores of this CalculatedMetricValues.

        Confidence score from 0 (no idea) to 100 (very confident). The same order as `metrics`.

        :param confidence_scores: The confidence scores of this CalculatedMetricValues.
        :type confidence_scores: List[int]
        """
        if confidence_scores is None:
            raise ValueError("Invalid value for `confidence_scores`, must not be `None`")

        self._confidence_scores = confidence_scores
