from datetime import timedelta

from athenian.api.models.web.base_model_ import Model
from athenian.api.typing_utils import VerbatimOptional


class CodeCheckRunStatistics(Model):
    """Gathered statistics about a group of check runs."""

    count: int
    successes: int
    critical: bool
    mean_execution_time: VerbatimOptional[timedelta]
    stddev_execution_time: VerbatimOptional[timedelta]
    median_execution_time: VerbatimOptional[timedelta]
    skips: int
    flaky_count: int
    count_timeline: list[int]
    successes_timeline: list[int]
    mean_execution_time_timeline: list[timedelta]
    median_execution_time_timeline: list[timedelta]

    def validate_successes(self, successes: int) -> int:
        """Sets the successes of this CodeCheckRunStatistics.

        Number of successful executions with respect to `date_from` and `date_to`.

        :param successes: The successes of this CodeCheckRunStatistics.
        """
        if successes is None:
            raise ValueError("Invalid value for `successes`, must not be `None`")
        if successes < 0:
            raise ValueError(
                "Invalid value for `successes`, must be a value greater than or equal to `0`",
            )

        return successes

    def validate_skips(self, skips: int) -> int:
        """Sets the skips of this CodeCheckRunStatistics.

        Number of times this check run was skipped.

        :param skips: The skips of this CodeCheckRunStatistics.
        """
        if skips is None:
            raise ValueError("Invalid value for `skips`, must not be `None`")
        if skips < 0:
            raise ValueError(
                "Invalid value for `skips`, must be a value greater than or equal to `0`",
            )

        return skips

    def validate_flaky_count(self, flaky_count: int) -> int:
        """Sets the flaky_count of this CodeCheckRunStatistics.

        Number of times this check run appeared flaky: it both failed and succeeded for the same
        commit.

        :param flaky_count: The flaky_count of this CodeCheckRunStatistics.
        """
        if flaky_count is None:
            raise ValueError("Invalid value for `flaky_count`, must not be `None`")
        if flaky_count < 0:
            raise ValueError(
                "Invalid value for `flaky_count`, must be a value greater than or equal to `0`",
            )

        return flaky_count
