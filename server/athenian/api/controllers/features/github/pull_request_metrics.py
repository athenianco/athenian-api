from datetime import datetime, timedelta
from typing import Optional

from athenian.api.controllers.features.github.pull_request import \
    PullRequestAverageMetricCalculator, PullRequestCounter, PullRequestMetricCalculator, \
    PullRequestSumMetricCalculator, register_metric
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.miners.types import PullRequestFacts
from athenian.api.models.web import PullRequestMetricID


@register_metric(PullRequestMetricID.PR_WIP_TIME)
class WorkInProgressTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time of work in progress metric."""

    may_have_negative_values = False

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                override_event_time: Optional[datetime] = None) -> Optional[timedelta]:
        """Calculate the actual state update."""
        if override_event_time is not None:
            wip_end = override_event_time
        elif facts.last_review:
            wip_end = facts.first_review_request.best
        else:
            # review was probably requested but never happened
            if facts.last_commit:
                wip_end = facts.last_commit.best
            else:
                # 0 commits in the PR, no reviews and review requests
                # => review time = 0
                # => merge time = 0 (you cannot merge an empty PR)
                # => release time = 0
                # This PR is 100% closed.
                wip_end = facts.closed.best
        if wip_end is not None and min_time <= wip_end < max_time:
            return wip_end - facts.work_began.best
        return None


@register_metric(PullRequestMetricID.PR_WIP_COUNT)
class WorkInProgressCounter(PullRequestCounter):
    """Count the number of PRs that were used to calculate PR_WIP_TIME."""

    calc_cls = WorkInProgressTimeCalculator


@register_metric(PullRequestMetricID.PR_REVIEW_TIME)
class ReviewTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time of the review process metric."""

    may_have_negative_values = False

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                allow_unclosed=False, override_event_time: Optional[datetime] = None,
                ) -> Optional[timedelta]:
        """Calculate the actual state update."""
        if not facts.first_review_request:
            return None
        if override_event_time is not None and min_time <= override_event_time < max_time:
            return override_event_time - facts.first_review_request.best
        # We cannot be sure that the approvals finished unless the PR is closed.
        if (facts.closed or allow_unclosed) and (
                (facts.approved and min_time <= facts.approved.best < max_time)
                or  # noqa
                (facts.last_review and min_time <= facts.last_review.best < max_time)):
            if facts.approved:
                return facts.approved.best - facts.first_review_request.best
            elif facts.last_review:
                return facts.last_review.best - facts.first_review_request.best
            else:
                return None
        return None


@register_metric(PullRequestMetricID.PR_REVIEW_COUNT)
class ReviewCounter(PullRequestCounter):
    """Count the number of PRs that were used to calculate PR_REVIEW_TIME."""

    calc_cls = ReviewTimeCalculator


@register_metric(PullRequestMetricID.PR_MERGING_TIME)
class MergingTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time to merge after finishing the review metric."""

    may_have_negative_values = False

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                override_event_time: Optional[datetime] = None) -> Optional[timedelta]:
        """Calculate the actual state update."""
        closed = facts.closed.best if override_event_time is None else override_event_time
        if closed is not None and min_time <= closed < max_time:
            # closed may mean either merged or not merged
            if facts.approved:
                return closed - facts.approved.best
            elif facts.last_review:
                return closed - facts.last_review.best
            elif facts.last_commit:
                return closed - facts.last_commit.best
        return None


@register_metric(PullRequestMetricID.PR_MERGING_COUNT)
class MergingCounter(PullRequestCounter):
    """Count the number of PRs that were used to calculate PR_MERGING_TIME."""

    calc_cls = MergingTimeCalculator


@register_metric(PullRequestMetricID.PR_RELEASE_TIME)
class ReleaseTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time to appear in a release after merging metric."""

    may_have_negative_values = False

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                override_event_time: Optional[datetime] = None) -> Optional[timedelta]:
        """Calculate the actual state update."""
        released = facts.released.best if override_event_time is None else override_event_time
        if facts.merged and released is not None and min_time <= released < max_time:
            return released - facts.merged.best
        return None


@register_metric(PullRequestMetricID.PR_RELEASE_COUNT)
class ReleaseCounter(PullRequestCounter):
    """Count the number of PRs that were used to calculate PR_RELEASE_TIME."""

    calc_cls = ReleaseTimeCalculator


@register_metric(PullRequestMetricID.PR_LEAD_TIME)
class LeadTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time to appear in a release since starting working on the PR."""

    may_have_negative_values = False

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                ) -> Optional[timedelta]:
        """Calculate the actual state update."""
        if facts.released and min_time <= facts.released.best < max_time:
            return facts.released.best - facts.work_began.best
        return None


@register_metric(PullRequestMetricID.PR_LEAD_COUNT)
class LeadCounter(PullRequestCounter):
    """Count the number of PRs that were used to calculate PR_LEAD_TIME."""

    calc_cls = LeadTimeCalculator


@register_metric(PullRequestMetricID.PR_CYCLE_TIME)
class CycleTimeCalculator(PullRequestMetricCalculator[timedelta]):
    """Sum of PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGE_TIME, and PR_RELEASE_TIME."""

    def __init__(self):
        """Initialize a new instance of CycleTimeCalculator."""
        super().__init__()
        self._calcs = [WorkInProgressTimeCalculator(), ReviewTimeCalculator(),
                       MergingTimeCalculator(), ReleaseTimeCalculator()]

    def value(self) -> Metric[timedelta]:
        """Calculate the current metric value."""
        exists = False
        ct = ct_conf_min = ct_conf_max = timedelta(0)
        for calc in self._calcs:
            val = calc.value()
            if val.exists:
                exists = True
                ct += val.value
                ct_conf_min += val.confidence_min
                ct_conf_max += val.confidence_max
        return Metric(exists, ct if exists else None,
                      ct_conf_min if exists else None, ct_conf_max if exists else None)

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                ) -> Optional[timedelta]:
        """Update the states of the underlying calcs and return whether at least one of the PR's \
        metrics exists."""
        exists = False
        sumval = timedelta(0)
        for calc in self._calcs:
            if calc(facts, min_time, max_time):
                exists = True
                sumval += calc.samples[-1]
        return sumval if exists else None

    def reset(self):
        """Reset the internal state."""
        for calc in self._calcs:
            calc.reset()


@register_metric(PullRequestMetricID.PR_CYCLE_COUNT)
class CycleCounter(PullRequestCounter):
    """Count unique PRs that were used to calculate PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGE_TIME, \
    or PR_RELEASE_TIME."""

    calc_cls = CycleTimeCalculator


@register_metric(PullRequestMetricID.PR_ALL_COUNT)
class AllCounter(PullRequestSumMetricCalculator[int]):
    """Count all the PRs that are active in the given time interval."""

    requires_full_span = True

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                ) -> Optional[int]:
        """Calculate the actual state update."""
        pr_started = facts.created.best  # not `work_began.best`! It breaks granular measurements.
        cut_before_released = (
            pr_started < min_time and facts.released and facts.released.best < min_time
        )
        cut_before_rejected = (
            pr_started < min_time and facts.closed and not facts.merged
            and facts.closed.best < min_time
        )
        cut_after = pr_started >= max_time
        # see also: ENG-673
        cut_old_unreleased = (
            facts.merged and not facts.released and facts.merged.best < min_time
        )
        if not (cut_before_released or cut_before_rejected or cut_after or cut_old_unreleased):
            return 1
        return None


@register_metric(PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME)
class WaitFirstReviewTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Elapsed time between requesting the review for the first time and getting it."""

    may_have_negative_values = False

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                ) -> Optional[timedelta]:
        """Calculate the actual state update."""
        if facts.first_review_request and facts.first_comment_on_first_review and \
                min_time <= facts.first_comment_on_first_review.best < max_time:
            return facts.first_comment_on_first_review.best - facts.first_review_request.best
        return None


@register_metric(PullRequestMetricID.PR_WAIT_FIRST_REVIEW_COUNT)
class WaitFirstReviewCounter(PullRequestCounter):
    """Count PRs that were used to calculate PR_WAIT_FIRST_REVIEW_TIME."""

    calc_cls = WaitFirstReviewTimeCalculator


@register_metric(PullRequestMetricID.PR_OPENED)
class OpenedCalculator(PullRequestSumMetricCalculator[int]):
    """Number of open PRs."""

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                ) -> Optional[int]:
        """Calculate the actual state update."""
        if min_time <= facts.created.best < max_time:
            return 1
        return None


@register_metric(PullRequestMetricID.PR_MERGED)
class MergedCalculator(PullRequestSumMetricCalculator[int]):
    """Number of merged PRs."""

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                ) -> Optional[int]:
        """Calculate the actual state update."""
        if facts.merged and min_time <= facts.merged.best < max_time:
            return 1
        return None


@register_metric(PullRequestMetricID.PR_REJECTED)
class RejectedCalculator(PullRequestSumMetricCalculator[int]):
    """Number of rejected PRs."""

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                ) -> Optional[int]:
        """Calculate the actual state update."""
        if facts.closed and not facts.merged and min_time <= facts.closed.best < max_time:
            return 1
        return None


@register_metric(PullRequestMetricID.PR_CLOSED)
class ClosedCalculator(PullRequestSumMetricCalculator[int]):
    """Number of closed PRs."""

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                ) -> Optional[int]:
        """Calculate the actual state update."""
        if facts.closed and min_time <= facts.closed.best < max_time:
            return 1
        return None


@register_metric(PullRequestMetricID.PR_RELEASED)
class ReleasedCalculator(PullRequestSumMetricCalculator[int]):
    """Number of released PRs."""

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                ) -> Optional[int]:
        """Calculate the actual state update."""
        if facts.released and min_time <= facts.released.best < max_time:
            return 1
        return None


@register_metric(PullRequestMetricID.PR_FLOW_RATIO)
class FlowRatioCalculator(PullRequestMetricCalculator[float]):
    """PR flow ratio - opened / closed - calculator."""

    def __init__(self):
        """Initialize a new instance of FlowRatioCalculator."""
        super().__init__()
        self._opened = OpenedCalculator()
        self._closed = ClosedCalculator()

    def value(self) -> Metric[float]:
        """Calculate the current metric value."""
        opened = self._opened.value()
        closed = self._closed.value()
        if not closed.exists and not opened.exists:
            return Metric(False, None, None, None)
        # Why +1? See ENG-866
        val = ((opened.value or 0) + 1) / ((closed.value or 0) + 1)
        return Metric(True, val, None, None)

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                ) -> Optional[float]:
        """Calculate the actual state update."""
        self._opened(facts, min_time, max_time)
        self._closed(facts, min_time, max_time)
        return None

    def reset(self):
        """Reset the internal state."""
        self._opened.reset()
        self._closed.reset()


@register_metric(PullRequestMetricID.PR_SIZE)
class SizeCalculator(PullRequestAverageMetricCalculator[int]):
    """Elapsed time between requesting the review for the first time and getting it."""

    may_have_negative_values = False

    def __init__(self):
        """Initialize a new instance of SizeCalculator."""
        super().__init__()
        self.all = AllCounter()

    def analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                ) -> Optional[int]:
        """Calculate the actual state update."""
        if self.all.analyze(facts, min_time, max_time) is not None:
            return facts.size
        return None
