import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class CodeBypassingPRsMeasurement(Model):
    """Statistics about code pushed outside of pull requests in a certain time interval."""

    attribute_types = {
        "date": datetime.date,
        "bypassed_commits": int,
        "bypassed_lines": int,
        "total_commits": int,
        "total_lines": int,
    }

    attribute_map = {
        "date": "date",
        "bypassed_commits": "bypassed_commits",
        "bypassed_lines": "bypassed_lines",
        "total_commits": "total_commits",
        "total_lines": "total_lines",
    }

    def __init__(
        self,
        date: Optional[datetime.date] = None,
        bypassed_commits: Optional[int] = None,
        bypassed_lines: Optional[int] = None,
        total_commits: Optional[int] = None,
        total_lines: Optional[int] = None,
    ):
        """CodeBypassingPRsMeasurement - a model defined in OpenAPI

        :param date: The date of this CodeBypassingPRsMeasurement.
        :param bypassed_commits: The bypassed_commits of this CodeBypassingPRsMeasurement.
        :param bypassed_lines: The bypassed_lines of this CodeBypassingPRsMeasurement.
        :param total_commits: The total_commits of this CodeBypassingPRsMeasurement.
        :param total_lines: The total_lines of this CodeBypassingPRsMeasurement.
        """
        self._date = date
        self._bypassed_commits = bypassed_commits
        self._bypassed_lines = bypassed_lines
        self._total_commits = total_commits
        self._total_lines = total_lines

    @property
    def date(self) -> datetime.date:
        """Gets the date of this CodeBypassingPRsMeasurement.

        Commits were pushed beginning with this date. They were not pushed later than +granularity.

        :return: The date of this CodeBypassingPRsMeasurement.
        """
        return self._date

    @date.setter
    def date(self, date: datetime.date):
        """Sets the date of this CodeBypassingPRsMeasurement.

        Commits were pushed beginning with this date. They were not pushed later than +granularity.

        :param date: The date of this CodeBypassingPRsMeasurement.
        :type date: date
        """
        if date is None:
            raise ValueError("Invalid value for `date`, must not be `None`")

        self._date = date

    @property
    def bypassed_commits(self) -> int:
        """Gets the bypassed_commits of this CodeBypassingPRsMeasurement.

        Number of commits that bypassed PRs in the time interval.

        :return: The bypassed_commits of this CodeBypassingPRsMeasurement.
        """
        return self._bypassed_commits

    @bypassed_commits.setter
    def bypassed_commits(self, bypassed_commits: int):
        """Sets the bypassed_commits of this CodeBypassingPRsMeasurement.

        Number of commits that bypassed PRs in the time interval.

        :param bypassed_commits: The bypassed_commits of this CodeBypassingPRsMeasurement.
        """
        if bypassed_commits is None:
            raise ValueError("Invalid value for `bypassed_commits`, must not be `None`")

        self._bypassed_commits = bypassed_commits

    @property
    def bypassed_lines(self) -> int:
        """Gets the bypassed_lines of this CodeBypassingPRsMeasurement.

        Number of changed (added + removed) lines that bypassed PRs in the time interval.

        :return: The bypassed_lines of this CodeBypassingPRsMeasurement.
        """
        return self._bypassed_lines

    @bypassed_lines.setter
    def bypassed_lines(self, bypassed_lines: int):
        """Sets the bypassed_lines of this CodeBypassingPRsMeasurement.

        Number of changed (added + removed) lines that bypassed PRs in the time interval.

        :param bypassed_lines: The bypassed_lines of this CodeBypassingPRsMeasurement.
        """
        if bypassed_lines is None:
            raise ValueError("Invalid value for `bypassed_lines`, must not be `None`")

        self._bypassed_lines = bypassed_lines

    @property
    def total_commits(self) -> int:
        """Gets the total_commits of this CodeBypassingPRsMeasurement.

        Overall number of commits in the time interval.

        :return: The total_commits of this CodeBypassingPRsMeasurement.
        """
        return self._total_commits

    @total_commits.setter
    def total_commits(self, total_commits: int):
        """Sets the total_commits of this CodeBypassingPRsMeasurement.

        Overall number of commits in the time interval.

        :param total_commits: The total_commits of this CodeBypassingPRsMeasurement.
        """
        if total_commits is None:
            raise ValueError("Invalid value for `total_commits`, must not be `None`")

        self._total_commits = total_commits

    @property
    def total_lines(self) -> int:
        """Gets the total_lines of this CodeBypassingPRsMeasurement.

        Overall number of changed (added + removed) lines in the time interval.

        :return: The total_lines of this CodeBypassingPRsMeasurement.
        """
        return self._total_lines

    @total_lines.setter
    def total_lines(self, total_lines: int):
        """Sets the total_lines of this CodeBypassingPRsMeasurement.

        Overall number of changed (added + removed) lines in the time interval.

        :param total_lines: The total_lines of this CodeBypassingPRsMeasurement.
        """
        self._total_lines = total_lines
