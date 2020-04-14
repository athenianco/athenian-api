from datetime import datetime, timedelta
from typing import Optional

from athenian.api.controllers.features.github.pull_request import \
    PullRequestAverageMetricCalculator, PullRequestCounter, PullRequestMetricCalculator, \
    PullRequestSumMetricCalculator, register
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.miners.github.pull_request import PullRequestTimes
from athenian.api.models.web import PullRequestMetricID


@register(PullRequestMetricID.PR_WIP_TIME)
class WorkInProgressTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time of work in progress metric."""

    may_have_negative_values = False

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                override_event_time: Optional[datetime] = None) -> Optional[timedelta]:
        """Calculate the actual state update."""
        if override_event_time is not None:
            wip_end = override_event_time
        elif times.last_review:
            wip_end = times.first_review_request.best
        else:
            # review was probably requested but never happened
            wip_end = times.last_commit.best
        if wip_end is not None and min_time <= wip_end <= max_time:
            return wip_end - times.work_began.best
        return None


@register(PullRequestMetricID.PR_WIP_COUNT)
class WorkInProgressCounter(PullRequestCounter):
    """Count the number of PRs that were used to calculate PR_WIP_TIME."""

    calc_cls = WorkInProgressTimeCalculator


@register(PullRequestMetricID.PR_REVIEW_TIME)
class ReviewTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time of the review process metric."""

    may_have_negative_values = False

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                allow_unclosed=False, override_event_time: Optional[datetime] = None,
                ) -> Optional[timedelta]:
        """Calculate the actual state update."""
        if not times.first_review_request:
            return None
        if override_event_time is not None and min_time <= override_event_time <= max_time:
            return override_event_time - times.first_review_request.best
        # We cannot be sure that the approvals finished unless the PR is closed.
        if (times.closed or allow_unclosed) and (
                (times.approved and min_time <= times.approved.best <= max_time)
                or  # noqa
                (times.last_review and min_time <= times.last_review.best <= max_time)):
            if times.approved:
                return times.approved.best - times.first_review_request.best
            elif times.last_review:
                return times.last_review.best - times.first_review_request.best
            else:
                return None
        return None


@register(PullRequestMetricID.PR_REVIEW_COUNT)
class ReviewCounter(PullRequestCounter):
    """Count the number of PRs that were used to calculate PR_REVIEW_TIME."""

    calc_cls = ReviewTimeCalculator


@register(PullRequestMetricID.PR_MERGING_TIME)
class MergingTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time to merge after finishing the review metric."""

    may_have_negative_values = False

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                override_event_time: Optional[datetime] = None) -> Optional[timedelta]:
        """Calculate the actual state update."""
        closed = times.closed.best if override_event_time is None else override_event_time
        if closed is not None and min_time <= closed <= max_time:
            # closed may mean either merged or not merged
            if times.approved:
                return closed - times.approved.best
            elif times.last_review:
                return closed - times.last_review.best
            elif times.last_commit:
                return closed - times.last_commit.best
        return None


@register(PullRequestMetricID.PR_MERGING_COUNT)
class MergingCounter(PullRequestCounter):
    """Count the number of PRs that were used to calculate PR_MERGING_TIME."""

    calc_cls = MergingTimeCalculator


@register(PullRequestMetricID.PR_RELEASE_TIME)
class ReleaseTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time to appear in a release after merging metric."""

    may_have_negative_values = False

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                override_event_time: Optional[datetime] = None) -> Optional[timedelta]:
        """Calculate the actual state update."""
        released = times.released.best if override_event_time is None else override_event_time
        if times.merged and released is not None and min_time <= released <= max_time:
            return released - times.merged.best
        return None


@register(PullRequestMetricID.PR_RELEASE_COUNT)
class ReleaseCounter(PullRequestCounter):
    """Count the number of PRs that were used to calculate PR_RELEASE_TIME."""

    calc_cls = ReleaseTimeCalculator


@register(PullRequestMetricID.PR_LEAD_TIME)
class LeadTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time to appear in a release since starting working on the PR."""

    may_have_negative_values = False

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[timedelta]:
        """Calculate the actual state update."""
        if times.released and min_time <= times.released.best <= max_time:
            return times.released.best - times.work_began.best
        return None


@register(PullRequestMetricID.PR_LEAD_COUNT)
class LeadCounter(PullRequestCounter):
    """Count the number of PRs that were used to calculate PR_LEAD_TIME."""

    calc_cls = LeadTimeCalculator


@register(PullRequestMetricID.PR_CYCLE_TIME)
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

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[timedelta]:
        """Update the states of the underlying calcs and return whether at least one of the PR's \
        metrics exists."""
        exists = False
        for calc in self._calcs:
            exists |= calc(times, min_time, max_time)
        return timedelta(0) if exists else None

    def reset(self):
        """Reset the internal state."""
        for calc in self._calcs:
            calc.reset()


@register(PullRequestMetricID.PR_CYCLE_COUNT)
class CycleCounter(PullRequestCounter):
    """Count unique PRs that were used to calculate PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGE_TIME, \
    and PR_RELEASE_TIME."""

    calc_cls = CycleTimeCalculator


@register(PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME)
class WaitFirstReviewTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Elapsed time between requesting the review for the first time and getting it."""

    may_have_negative_values = False

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[timedelta]:
        """Calculate the actual state update."""
        if times.first_review_request and times.first_comment_on_first_review and \
                min_time <= times.first_comment_on_first_review.best <= max_time:
            return times.first_comment_on_first_review.best - times.first_review_request.best
        return None


@register(PullRequestMetricID.PR_OPENED)
class OpenedCalculator(PullRequestSumMetricCalculator[int]):
    """Number of open PRs."""

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[int]:
        """Calculate the actual state update."""
        if min_time <= times.created.best <= max_time:
            return 1
        return None


@register(PullRequestMetricID.PR_MERGED)
class MergedCalculator(PullRequestSumMetricCalculator[int]):
    """Number of merged PRs."""

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[int]:
        """Calculate the actual state update."""
        if times.merged and min_time <= times.merged.best <= max_time:
            return 1
        return None


@register(PullRequestMetricID.PR_CLOSED)
class ClosedCalculator(PullRequestSumMetricCalculator[int]):
    """Number of closed PRs."""

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[int]:
        """Calculate the actual state update."""
        if times.closed and min_time <= times.closed.best <= max_time:
            return 1
        return None


@register(PullRequestMetricID.PR_RELEASED)
class ReleasedCalculator(PullRequestSumMetricCalculator[int]):
    """Number of released PRs."""

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[int]:
        """Calculate the actual state update."""
        if times.released and min_time <= times.released.best <= max_time:
            return 1
        return None


@register(PullRequestMetricID.PR_FLOW_RATIO)
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
        if not closed.exists:
            return Metric(False, None, None, None)
        if not opened.exists:
            return Metric(True, 0, None, None)  # yes, it is True, not False as above
        val = opened.value / closed.value
        return Metric(True, val, None, None)

    def analyze(self, times: PullRequestTimes, min_time: datetime, max_time: datetime,
                ) -> Optional[float]:
        """Calculate the actual state update."""
        self._opened(times, min_time, max_time)
        self._closed(times, min_time, max_time)
        return None

    def reset(self):
        """Reset the internal state."""
        self._opened.reset()
        self._closed.reset()
