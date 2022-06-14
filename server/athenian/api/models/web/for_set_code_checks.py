from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.for_set_common import CommonPullRequestFilters, ForSetLines


class _ForSetCodeChecks(Model, sealed=False):
    """Filters for `/metrics/code_checks` and `/histograms/code_checks`."""

    attribute_types = {
        "pushers": Optional[List[str]],
        "pusher_groups": Optional[List[List[str]]],
    }

    attribute_map = {
        "pushers": "pushers",
        "pusher_groups": "pusher_groups",
    }

    def __init__(
        self,
        pushers: Optional[List[str]] = None,
        pusher_groups: Optional[List[List[str]]] = None,
    ):
        """ForSetCodeChecks - a model defined in OpenAPI

        :param pushers: The pushers of this ForSetCodeChecks.
        :param pusher_groups: The pusher_groups of this ForSetCodeChecks.
        """
        self._pushers = pushers
        self._pusher_groups = pusher_groups

    @property
    def pushers(self) -> Optional[List[str]]:
        """Gets the pushers of this ForSetCodeChecks.

        Check runs must be triggered by commits pushed by these people.

        :return: The pushers of this ForSetCodeChecks.
        """
        return self._pushers

    @pushers.setter
    def pushers(self, pushers: Optional[List[str]]):
        """Sets the pushers of this ForSetCodeChecks.

        Check runs must be triggered by commits pushed by these people.

        :param pushers: The pushers of this ForSetCodeChecks.
        """
        self._pushers = pushers

    @property
    def pusher_groups(self) -> Optional[List[List[str]]]:
        """Gets the pusher_groups of this ForSetCodeChecks.

        Check runs must be triggered by commits authored by these people. We aggregate by each
        group so that you can request metrics of several teams at once. We treat `pushers`
        as another group, if specified.

        :return: The pusher_groups of this ForSetCodeChecks.
        """
        return self._pusher_groups

    @pusher_groups.setter
    def pusher_groups(self, pusher_groups: Optional[List[List[str]]]):
        """Sets the pusher_groups of this ForSetCodeChecks.

        Check runs must be triggered by commits authored by these people. We aggregate by each
        group so that you can request metrics of several teams at once. We treat `pushers`
        as another group, if specified.

        :param pusher_groups: The pusher_groups of this ForSetCodeChecks.
        """
        self._pusher_groups = pusher_groups

    def select_pushers_group(self, index: int) -> "ForSetCodeChecks":
        """Change `pushers` to point at the specified `pushers_group`."""
        fs = self.copy()
        if self.pusher_groups is None:
            if index > 0:
                raise IndexError("%d is out of range (no pusher_groups)" % index)
            return fs
        if index >= len(self.pusher_groups):
            raise IndexError(
                "%d is out of range (max is %d)" % (index, len(self.withpusher_groupsgroups) - 1),
            )
        fs.pushers = self.pusher_groups[index]
        fs.pusher_groups = None
        return fs


ForSetCodeChecks = AllOf(
    _ForSetCodeChecks,
    ForSetLines,
    CommonPullRequestFilters,
    name="ForSetCodeChecks",
    module=__name__,
)


class _CalculatedCodeCheckCommon(Model, sealed=False):
    attribute_types = {
        "for_": ForSetCodeChecks,
        "check_runs": Optional[int],
        "suites_ratio": Optional[float],
    }

    attribute_map = {
        "for_": "for",
        "check_runs": "check_runs",
        "suites_ratio": "suites_ratio",
    }

    def __init__(
        self,
        for_: Optional[ForSetCodeChecks] = None,
        check_runs: Optional[int] = None,
        suites_ratio: Optional[float] = None,
    ):
        """CalculatedCodeCheckMetricsItem - a model defined in OpenAPI

        :param for_: The for_ of this CalculatedCodeCheckMetricsItem.
        :param check_runs: The check_runs of this CalculatedCodeCheckMetricsItem.
        :param suites_ratio: The suites_ratio of this CalculatedCodeCheckMetricsItem.
        """
        self._for_ = for_
        self._check_runs = check_runs
        self._suites_ratio = suites_ratio

    @property
    def for_(self) -> ForSetCodeChecks:
        """Gets the for_ of this CalculatedCodeCheckMetricsItem.

        :return: The for_ of this CalculatedCodeCheckMetricsItem.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: ForSetCodeChecks):
        """Sets the for_ of this CalculatedCodeCheckMetricsItem.

        :param for_: The for_ of this CalculatedCodeCheckMetricsItem.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for_`, must not be `None`")

        self._for_ = for_

    @property
    def check_runs(self) -> Optional[int]:
        """Gets the check_runs of this CalculatedCodeCheckMetricsItem.

        We calculated metrics for check suites with this number of runs. Not null only if the user
        specified `split_by_check_runs = true`.

        :return: The check_runs of this CalculatedCodeCheckMetricsItem.
        """
        return self._check_runs

    @check_runs.setter
    def check_runs(self, check_runs: Optional[int]):
        """Sets the check_runs of this CalculatedCodeCheckMetricsItem.

        We calculated metrics for check suites with this number of runs. Not null only if the user
        specified `split_by_check_runs = true`.

        :param check_runs: The check_runs of this CalculatedCodeCheckMetricsItem.
        """
        self._check_runs = check_runs

    @property
    def suites_ratio(self) -> Optional[float]:
        """Gets the suites_ratio of this CalculatedCodeCheckMetricsItem.

        Number of check suites with `check_runs` number of check runs divided by the overall number
        of check suites. Not null only if the user specified `split_by_check_runs = true`.

        :return: The suites_ratio of this CalculatedCodeCheckMetricsItem.
        """
        return self._suites_ratio

    @suites_ratio.setter
    def suites_ratio(self, suites_ratio: Optional[float]):
        """Sets the suites_ratio of this CalculatedCodeCheckMetricsItem.

        Number of check suites with `check_runs` number of check runs divided by the overall number
        of check suites. Not null only if the user specified `split_by_check_runs = true`.

        :param suites_ratio: The suites_ratio of this CalculatedCodeCheckMetricsItem.
        """
        if suites_ratio is not None and suites_ratio > 1:
            raise ValueError(
                "Invalid value for `suites_ratio`, must be a value less than or equal to `1`",
            )
        if suites_ratio is not None and suites_ratio < 0:
            raise ValueError(
                "Invalid value for `suites_ratio`, must be a value greater than or equal to `0`",
            )

        self._suites_ratio = suites_ratio
