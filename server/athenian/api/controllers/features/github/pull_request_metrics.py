from datetime import timedelta
from typing import Optional

from athenian.api.controllers.features.github.pull_request import \
    PullRequestAverageMetricCalculator, register
from athenian.api.controllers.miners.github.pull_request import PullRequestTimes


@register("pr-wip-time")
class WorkInProgressTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time of work in progress metric."""

    def analyze(self, times: PullRequestTimes) -> Optional[timedelta]:
        """Do the actual state update. See the design document for more info."""
        if times.first_review_request.best is not None:
            return times.first_review_request.best - times.work_begins.best
        return None


@register("pr-wait-first-review-time")
class WaitFirstReviewTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time of waiting for the first review metric."""

    def analyze(self, times: PullRequestTimes) -> Optional[timedelta]:
        """Do the actual state update. See the design document for more info."""
        if times.first_comment_on_first_review.best is not None and \
                times.first_review_request.best is not None:
            return times.first_comment_on_first_review.best - times.first_review_request.best
        return None


@register("pr-review-time")
class ReviewTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time of the review process metric."""

    def analyze(self, times: PullRequestTimes) -> Optional[timedelta]:
        """Do the actual state update. See the design document for more info."""
        if times.first_review_request.best is not None and times.approved.best is not None:
            return times.approved.best - times.first_review_request.best
        return None


@register("pr-merging-time")
class MergeTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time to merge after finishing the review metric."""

    def analyze(self, times: PullRequestTimes) -> Optional[timedelta]:
        """Do the actual state update. See the design document for more info."""
        if times.approved.value is not None and times.merged.value is not None:
            return times.merged.value - times.approved.value
        return None


@register("pr-release-time")
class ReleaseTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time to appear in a release after merging metric."""

    def analyze(self, times: PullRequestTimes) -> Optional[timedelta]:
        """Do the actual state update. See the design document for more info."""
        if times.merged.value is not None and times.released.value is not None:
            return times.released.value - times.merged.value
        return None


@register("pr-lead-time")
class LeadTimeCalculator(PullRequestAverageMetricCalculator[timedelta]):
    """Time to appear in a release since starting working on the PR."""

    def analyze(self, times: PullRequestTimes) -> Optional[timedelta]:
        """Do the actual state update. See the design document for more info."""
        if times.released.value is not None:
            return times.released.value - times.work_begins.best
        return None
