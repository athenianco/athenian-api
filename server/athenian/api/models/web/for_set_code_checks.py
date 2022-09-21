from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.for_set_common import CommonPullRequestFilters, ForSetLines


class _ForSetCodeChecks(Model, sealed=False):
    """Filters for `/metrics/code_checks` and `/histograms/code_checks`."""

    pushers: Optional[List[str]]
    pusher_groups: Optional[List[List[str]]]

    def select_pushers_group(self, index: int) -> "ForSetCodeChecks":
        """Change `pushers` to point at the specified `pushers_group`."""
        fs = self.copy()
        if self.pusher_groups is None:
            if index > 0:
                raise IndexError("%d is out of range (no pusher_groups)" % index)
            return fs
        if index >= len(self.pusher_groups):
            raise IndexError(
                "%d is out of range (max is %d)" % (index, len(self.pusher_groups) - 1),
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
    for_: (ForSetCodeChecks, "for")
    check_runs: Optional[int]
    suites_ratio: Optional[float]

    def validate_suites_ratio(self, suites_ratio: Optional[float]) -> Optional[float]:
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

        return suites_ratio
