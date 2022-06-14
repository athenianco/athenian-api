from datetime import timedelta
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class CodeCheckRunStatistics(Model):
    """Gathered statistics about a group of check runs."""

    attribute_types = {
        "count": int,
        "successes": int,
        "critical": bool,
        "mean_execution_time": timedelta,
        "stddev_execution_time": timedelta,
        "median_execution_time": timedelta,
        "skips": int,
        "flaky_count": int,
        "count_timeline": List[int],
        "successes_timeline": List[int],
        "mean_execution_time_timeline": List[timedelta],
        "median_execution_time_timeline": List[timedelta],
    }

    attribute_map = {
        "count": "count",
        "successes": "successes",
        "critical": "critical",
        "mean_execution_time": "mean_execution_time",
        "stddev_execution_time": "stddev_execution_time",
        "median_execution_time": "median_execution_time",
        "skips": "skips",
        "flaky_count": "flaky_count",
        "count_timeline": "count_timeline",
        "successes_timeline": "successes_timeline",
        "mean_execution_time_timeline": "mean_execution_time_timeline",
        "median_execution_time_timeline": "median_execution_time_timeline",
    }

    def __init__(
        self,
        count: Optional[int] = None,
        successes: Optional[int] = None,
        critical: Optional[bool] = None,
        mean_execution_time: Optional[timedelta] = None,
        stddev_execution_time: Optional[timedelta] = None,
        median_execution_time: Optional[timedelta] = None,
        skips: Optional[int] = None,
        flaky_count: Optional[int] = None,
        count_timeline: Optional[List[int]] = None,
        successes_timeline: Optional[List[int]] = None,
        mean_execution_time_timeline: Optional[List[timedelta]] = None,
        median_execution_time_timeline: Optional[List[timedelta]] = None,
    ):
        """CodeCheckRunStatistics - a model defined in OpenAPI

        :param count: The count of this CodeCheckRunStatistics.
        :param successes: The successes of this CodeCheckRunStatistics.
        :param critical: The critical of this CodeCheckRunStatistics.
        :param mean_execution_time: The mean_execution_time of this CodeCheckRunStatistics.
        :param median_execution_time: The median_execution_time of this CodeCheckRunStatistics.
        :param skips: The skips of this CodeCheckRunStatistics.
        :param flaky_count: The flaky_count of this CodeCheckRunStatistics.
        :param count_timeline: The count_timeline of this CodeCheckRunStatistics.
        :param successes_timeline: The successes_timeline of this CodeCheckRunStatistics.
        :param mean_execution_time_timeline: The mean_execution_time_timeline of this \
               CodeCheckRunStatistics.
        :param stddev_execution_time: The stddev_execution_time of this CodeCheckRunStatistics.
        :param median_execution_time_timeline: The median_execution_time_timeline of this \
               CodeCheckRunStatistics.
        """
        self._count = count
        self._successes = successes
        self._critical = critical
        self._mean_execution_time = mean_execution_time
        self._stddev_execution_time = stddev_execution_time
        self._median_execution_time = median_execution_time
        self._skips = skips
        self._flaky_count = flaky_count
        self._count_timeline = count_timeline
        self._successes_timeline = successes_timeline
        self._mean_execution_time_timeline = mean_execution_time_timeline
        self._median_execution_time_timeline = median_execution_time_timeline

    @property
    def count(self) -> int:
        """Gets the count of this CodeCheckRunStatistics.

        Number of executions with respect to `date_from` and `date_to`.

        :return: The count of this CodeCheckRunStatistics.
        """
        return self._count

    @count.setter
    def count(self, count: int):
        """Sets the count of this CodeCheckRunStatistics.

        Number of executions with respect to `date_from` and `date_to`.

        :param count: The count of this CodeCheckRunStatistics.
        """
        if count is None:
            raise ValueError("Invalid value for `count`, must not be `None`")
        if count < 1:
            raise ValueError(
                "Invalid value for `count`, must be a value greater than or equal to `1`"
            )

        self._count = count

    @property
    def successes(self) -> int:
        """Gets the successes of this CodeCheckRunStatistics.

        Number of successful executions with respect to `date_from` and `date_to`.

        :return: The successes of this CodeCheckRunStatistics.
        """
        return self._successes

    @successes.setter
    def successes(self, successes: int):
        """Sets the successes of this CodeCheckRunStatistics.

        Number of successful executions with respect to `date_from` and `date_to`.

        :param successes: The successes of this CodeCheckRunStatistics.
        """
        if successes is None:
            raise ValueError("Invalid value for `successes`, must not be `None`")
        if successes < 0:
            raise ValueError(
                "Invalid value for `successes`, must be a value greater than or equal to `0`"
            )

        self._successes = successes

    @property
    def critical(self) -> int:
        """Gets the critical of this CodeCheckRunStatistics.

        Number of successful executions with respect to `date_from` and `date_to`.

        :return: The critical of this CodeCheckRunStatistics.
        """
        return self._critical

    @critical.setter
    def critical(self, critical: int):
        """Sets the critical of this CodeCheckRunStatistics.

        Number of successful executions with respect to `date_from` and `date_to`.

        :param critical: The critical of this CodeCheckRunStatistics.
        """
        if critical is None:
            raise ValueError("Invalid value for `critical`, must not be `None`")

        self._critical = critical

    @property
    def mean_execution_time(self) -> timedelta:
        """Gets the mean_execution_time of this CodeCheckRunStatistics.

        :return: The mean_execution_time of this CodeCheckRunStatistics.
        """
        return self._mean_execution_time

    @mean_execution_time.setter
    def mean_execution_time(self, mean_execution_time: timedelta):
        """Sets the mean_execution_time of this CodeCheckRunStatistics.

        :param mean_execution_time: The mean_execution_time of this CodeCheckRunStatistics.
        """
        self._mean_execution_time = mean_execution_time

    @property
    def stddev_execution_time(self) -> timedelta:
        """Gets the stddev_execution_time of this CodeCheckRunStatistics.

        :return: The stddev_execution_time of this CodeCheckRunStatistics.
        """
        return self._stddev_execution_time

    @stddev_execution_time.setter
    def stddev_execution_time(self, stddev_execution_time: timedelta):
        """Sets the stddev_execution_time of this CodeCheckRunStatistics.

        :param stddev_execution_time: The stddev_execution_time of this CodeCheckRunStatistics.
        """
        self._stddev_execution_time = stddev_execution_time

    @property
    def median_execution_time(self) -> timedelta:
        """Gets the median_execution_time of this CodeCheckRunStatistics.

        :return: The median_execution_time of this CodeCheckRunStatistics.
        """
        return self._median_execution_time

    @median_execution_time.setter
    def median_execution_time(self, median_execution_time: timedelta):
        """Sets the median_execution_time of this CodeCheckRunStatistics.

        :param median_execution_time: The median_execution_time of this CodeCheckRunStatistics.
        """
        self._median_execution_time = median_execution_time

    @property
    def skips(self) -> int:
        """Gets the skips of this CodeCheckRunStatistics.

        Number of times this check run was skipped.

        :return: The skips of this CodeCheckRunStatistics.
        """
        return self._skips

    @skips.setter
    def skips(self, skips: int):
        """Sets the skips of this CodeCheckRunStatistics.

        Number of times this check run was skipped.

        :param skips: The skips of this CodeCheckRunStatistics.
        """
        if skips is None:
            raise ValueError("Invalid value for `skips`, must not be `None`")
        if skips < 0:
            raise ValueError(
                "Invalid value for `skips`, must be a value greater than or equal to `0`"
            )

        self._skips = skips

    @property
    def flaky_count(self) -> int:
        """Gets the flaky_count of this CodeCheckRunStatistics.

        Number of times this check run appeared flaky: it both failed and succeeded for the same
        commit.

        :return: The flaky_count of this CodeCheckRunStatistics.
        """
        return self._flaky_count

    @flaky_count.setter
    def flaky_count(self, flaky_count: int):
        """Sets the flaky_count of this CodeCheckRunStatistics.

        Number of times this check run appeared flaky: it both failed and succeeded for the same
        commit.

        :param flaky_count: The flaky_count of this CodeCheckRunStatistics.
        """
        if flaky_count is None:
            raise ValueError("Invalid value for `flaky_count`, must not be `None`")
        if flaky_count < 0:
            raise ValueError(
                "Invalid value for `flaky_count`, must be a value greater than or equal to `0`"
            )

        self._flaky_count = flaky_count

    @property
    def count_timeline(self) -> List[int]:
        """Gets the count_timeline of this CodeCheckRunStatistics.

        Number of executions through time. The dates sequence is `FilteredCodeCheckRuns.timeline`.

        :return: The count_timeline of this CodeCheckRunStatistics.
        """
        return self._count_timeline

    @count_timeline.setter
    def count_timeline(self, count_timeline: List[int]):
        """Sets the count_timeline of this CodeCheckRunStatistics.

        Number of executions through time. The dates sequence is `FilteredCodeCheckRuns.timeline`.

        :param count_timeline: The count_timeline of this CodeCheckRunStatistics.
        """
        if count_timeline is None:
            raise ValueError("Invalid value for `count_timeline`, must not be `None`")

        self._count_timeline = count_timeline

    @property
    def successes_timeline(self) -> List[int]:
        """Gets the successes_timeline of this CodeCheckRunStatistics.

        Number of successful executions through time. The dates sequence is
        `FilteredCodeCheckRuns.timeline`.

        :return: The successes_timeline of this CodeCheckRunStatistics.
        """
        return self._successes_timeline

    @successes_timeline.setter
    def successes_timeline(self, successes_timeline: List[int]):
        """Sets the successes_timeline of this CodeCheckRunStatistics.

        Number of successful executions through time. The dates sequence is
        `FilteredCodeCheckRuns.timeline`.

        :param successes_timeline: The successes_timeline of this CodeCheckRunStatistics.
        """
        if successes_timeline is None:
            raise ValueError("Invalid value for `successes_timeline`, must not be `None`")

        self._successes_timeline = successes_timeline

    @property
    def mean_execution_time_timeline(self) -> List[timedelta]:
        """Gets the mean_execution_time_timeline of this CodeCheckRunStatistics.

        Average elapsed execution time through time. The dates sequence is
        `FilteredCodeCheckRuns.timeline`.

        :return: The mean_execution_time_timeline of this CodeCheckRunStatistics.
        """
        return self._mean_execution_time_timeline

    @mean_execution_time_timeline.setter
    def mean_execution_time_timeline(self, mean_execution_time_timeline: List[timedelta]):
        """Sets the mean_execution_time_timeline of this CodeCheckRunStatistics.

        Average elapsed execution time through time. The dates sequence is
        `FilteredCodeCheckRuns.timeline`.

        :param mean_execution_time_timeline: The mean_execution_time_timeline of this \
                                             CodeCheckRunStatistics.
        """
        if mean_execution_time_timeline is None:
            raise ValueError(
                "Invalid value for `mean_execution_time_timeline`, must not be `None`"
            )

        self._mean_execution_time_timeline = mean_execution_time_timeline

    @property
    def median_execution_time_timeline(self) -> List[timedelta]:
        """Gets the median_execution_time_timeline of this CodeCheckRunStatistics.

        Median elapsed execution time through time. The dates sequence is
        `FilteredCodeCheckRuns.timeline`.

        :return: The median_execution_time_timeline of this CodeCheckRunStatistics.
        """
        return self._median_execution_time_timeline

    @median_execution_time_timeline.setter
    def median_execution_time_timeline(self, median_execution_time_timeline: List[timedelta]):
        """Sets the median_execution_time_timeline of this CodeCheckRunStatistics.

        Median elapsed execution time through time. The dates sequence is
        `FilteredCodeCheckRuns.timeline`.

        :param median_execution_time_timeline: The median_execution_time_timeline of this \
                                               CodeCheckRunStatistics.
        """
        if median_execution_time_timeline is None:
            raise ValueError(
                "Invalid value for `median_execution_time_timeline`, must not be `None`"
            )

        self._median_execution_time_timeline = median_execution_time_timeline
