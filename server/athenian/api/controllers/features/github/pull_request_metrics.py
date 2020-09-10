from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence, Type

import numpy as np

from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    BinnedMetricCalculator, Counter, CounterWithQuantiles, HistogramCalculator, \
    HistogramCalculatorEnsemble, \
    MetricCalculator, MetricCalculatorEnsemble, SumMetricCalculator
from athenian.api.controllers.miners.types import PullRequestFacts, T
from athenian.api.models.web import PullRequestMetricID


metric_calculators: Dict[str, Type[MetricCalculator]] = {}
histogram_calculators: Dict[str, Type[HistogramCalculator]] = {}


class PullRequestMetricCalculatorEnsemble(MetricCalculatorEnsemble[T]):
    """MetricCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of PullRequestMetricCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=metric_calculators)


class PullRequestHistogramCalculatorEnsemble(HistogramCalculatorEnsemble[T]):
    """HistogramCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of PullRequestHistogramCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=histogram_calculators)


class PullRequestBinnedMetricCalculator(BinnedMetricCalculator[T]):
    """BinnedMetricCalculator adapted for pull requests."""

    def __init__(self,
                 metrics: Sequence[str],
                 time_intervals: Sequence[datetime],
                 quantiles: Sequence[float]):
        """Initialize a new instance of PullRequestBinnedMetricCalculator class."""
        super().__init__(metrics=metrics, time_intervals=time_intervals, quantiles=quantiles,
                         class_mapping=metric_calculators,
                         start_time_getter=lambda pr: pr.work_began,
                         finish_time_getter=lambda pr: pr.released)


def register_metric(name: str):
    """Keep track of the PR metric calculators and generate the histogram calculator."""
    assert isinstance(name, str)

    def register_with_name(cls: Type[MetricCalculator]):
        metric_calculators[name] = cls
        if not issubclass(cls, SumMetricCalculator):
            histogram_calculators[name] = \
                type("HistogramOf" + cls.__name__, (cls, HistogramCalculator), {})
        return cls

    return register_with_name


@register_metric(PullRequestMetricID.PR_WIP_TIME)
class WorkInProgressTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time of work in progress metric."""

    may_have_negative_values = False

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 override_event_time: Optional[datetime] = None) -> Optional[timedelta]:
        """Calculate the actual state update."""
        if override_event_time is not None:
            wip_end = override_event_time
        elif facts.last_review:
            wip_end = facts.first_review_request
        else:
            # review was probably requested but never happened
            if facts.last_commit:
                wip_end = facts.last_commit
            else:
                # 0 commits in the PR, no reviews and review requests
                # => review time = 0
                # => merge time = 0 (you cannot merge an empty PR)
                # => release time = 0
                # This PR is 100% closed.
                wip_end = facts.closed
        if wip_end and min_time <= wip_end < max_time:
            return wip_end - facts.work_began
        return None


@register_metric(PullRequestMetricID.PR_WIP_COUNT)
class WorkInProgressCounter(Counter):
    """Count the number of PRs that were used to calculate PR_WIP_TIME \
    disregarding the quantiles."""

    deps = (WorkInProgressTimeCalculator,)


@register_metric(PullRequestMetricID.PR_WIP_COUNT_Q)
class WorkInProgressCounterWithQuantiles(CounterWithQuantiles):
    """Count the number of PRs that were used to calculate PR_WIP_TIME respecting the quantiles."""

    deps = (WorkInProgressTimeCalculator,)


@register_metric(PullRequestMetricID.PR_REVIEW_TIME)
class ReviewTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time of the review process metric."""

    may_have_negative_values = False

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 allow_unclosed=False, override_event_time: Optional[datetime] = None,
                 ) -> Optional[timedelta]:
        """Calculate the actual state update."""
        if not facts.first_review_request:
            return None
        if override_event_time is not None and min_time <= override_event_time < max_time:
            return override_event_time - facts.first_review_request
        # We cannot be sure that the approvals finished unless the PR is closed.
        if (facts.closed or allow_unclosed) and (
                (facts.approved and min_time <= facts.approved < max_time)
                or  # noqa
                (facts.last_review and min_time <= facts.last_review < max_time)):
            if facts.approved:
                return facts.approved - facts.first_review_request
            elif facts.last_review:
                return facts.last_review - facts.first_review_request
            else:
                return None
        return None


@register_metric(PullRequestMetricID.PR_REVIEW_COUNT)
class ReviewCounter(Counter):
    """Count the number of PRs that were used to calculate PR_REVIEW_TIME disregarding \
    the quantiles."""

    deps = (ReviewTimeCalculator,)


@register_metric(PullRequestMetricID.PR_REVIEW_COUNT_Q)
class ReviewCounterWithQuantiles(CounterWithQuantiles):
    """Count the number of PRs that were used to calculate PR_REVIEW_TIME respecting \
    the quantiles."""

    deps = (ReviewTimeCalculator,)


@register_metric(PullRequestMetricID.PR_MERGING_TIME)
class MergingTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time to merge after finishing the review metric."""

    may_have_negative_values = False

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 override_event_time: Optional[datetime] = None) -> Optional[timedelta]:
        """Calculate the actual state update."""
        closed = facts.closed if override_event_time is None else override_event_time
        if closed is not None and min_time <= closed < max_time:
            # closed may mean either merged or not merged
            if facts.approved:
                return closed - facts.approved
            elif facts.last_review:
                return closed - facts.last_review
            elif facts.last_commit:
                return closed - facts.last_commit
        return None


@register_metric(PullRequestMetricID.PR_MERGING_COUNT)
class MergingCounter(Counter):
    """Count the number of PRs that were used to calculate PR_MERGING_TIME disregarding \
    the quantiles."""

    deps = (MergingTimeCalculator,)


@register_metric(PullRequestMetricID.PR_MERGING_COUNT_Q)
class MergingCounterWithQuantiles(CounterWithQuantiles):
    """Count the number of PRs that were used to calculate PR_MERGING_TIME respecting \
    the quantiles."""

    deps = (MergingTimeCalculator,)


@register_metric(PullRequestMetricID.PR_RELEASE_TIME)
class ReleaseTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time to appear in a release after merging metric."""

    may_have_negative_values = False

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 override_event_time: Optional[datetime] = None) -> Optional[timedelta]:
        """Calculate the actual state update."""
        released = facts.released if override_event_time is None else override_event_time
        if facts.merged and released is not None and min_time <= released < max_time:
            return released - facts.merged
        return None


@register_metric(PullRequestMetricID.PR_RELEASE_COUNT)
class ReleaseCounter(Counter):
    """Count the number of PRs that were used to calculate PR_RELEASE_TIME disregarding \
    the quantiles."""

    deps = (ReleaseTimeCalculator,)


@register_metric(PullRequestMetricID.PR_RELEASE_COUNT_Q)
class ReleaseCounterWithQuantiles(CounterWithQuantiles):
    """Count the number of PRs that were used to calculate PR_RELEASE_TIME respecting \
    the quantiles."""

    deps = (ReleaseTimeCalculator,)


@register_metric(PullRequestMetricID.PR_LEAD_TIME)
class LeadTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time to appear in a release since starting working on the PR."""

    may_have_negative_values = False

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[timedelta]:
        """Calculate the actual state update."""
        if facts.released and min_time <= facts.released < max_time:
            return facts.released - facts.work_began
        return None


@register_metric(PullRequestMetricID.PR_LEAD_COUNT)
class LeadCounter(Counter):
    """Count the number of PRs that were used to calculate PR_LEAD_TIME disregarding \
    the quantiles."""

    deps = (LeadTimeCalculator,)


@register_metric(PullRequestMetricID.PR_LEAD_COUNT_Q)
class LeadCounterWithQuantiles(CounterWithQuantiles):
    """Count the number of PRs that were used to calculate PR_LEAD_TIME respecting \
    the quantiles."""

    deps = (LeadTimeCalculator,)


@register_metric(PullRequestMetricID.PR_CYCLE_TIME)
class CycleTimeCalculator(MetricCalculator[timedelta]):
    """Sum of PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGE_TIME, and PR_RELEASE_TIME."""

    deps = (WorkInProgressTimeCalculator,
            ReviewTimeCalculator,
            MergingTimeCalculator,
            ReleaseTimeCalculator)

    def _value(self, samples: Sequence[timedelta]) -> Metric[timedelta]:
        """Calculate the current metric value."""
        exists = False
        ct = ct_conf_min = ct_conf_max = timedelta(0)
        for calc in self._calcs:
            val = calc.value
            if val.exists:
                exists = True
                ct += val.value
                ct_conf_min += val.confidence_min
                ct_conf_max += val.confidence_max
        return Metric(exists, ct if exists else None,
                      ct_conf_min if exists else None, ct_conf_max if exists else None)

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[timedelta]:
        """Update the states of the underlying calcs and return whether at least one of the PR's \
        metrics exists."""
        sumval = None
        for calc in self._calcs:
            peek = calc.peek
            if peek is not None:
                if sumval is None:
                    sumval = peek
                else:
                    sumval += peek
        return sumval

    def _cut_by_quantiles(self) -> np.ndarray:
        return self._asarray()


@register_metric(PullRequestMetricID.PR_CYCLE_COUNT)
class CycleCounter(Counter):
    """Count unique PRs that were used to calculate PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGE_TIME, \
    or PR_RELEASE_TIME disregarding the quantiles."""

    deps = (CycleTimeCalculator,)


@register_metric(PullRequestMetricID.PR_CYCLE_COUNT_Q)
class CycleCounterWithQuantiles(CounterWithQuantiles):
    """Count unique PRs that were used to calculate PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGE_TIME, \
    or PR_RELEASE_TIME respecting the quantiles."""

    deps = (CycleTimeCalculator,)


@register_metric(PullRequestMetricID.PR_ALL_COUNT)
class AllCounter(SumMetricCalculator[int]):
    """Count all the PRs that are active in the given time interval."""

    requires_full_span = True

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[int]:
        """Calculate the actual state update."""
        pr_started = facts.created  # not `work_began`! It breaks granular measurements.
        cut_before_released = (
            pr_started < min_time and facts.released and facts.released < min_time
        )
        cut_before_rejected = (
            pr_started < min_time and facts.closed and not facts.merged
            and facts.closed < min_time
        )
        cut_after = pr_started >= max_time
        # see also: ENG-673
        cut_old_unreleased = (
            facts.merged and not facts.released and facts.merged < min_time
        )
        if not (cut_before_released or cut_before_rejected or cut_after or cut_old_unreleased):
            return 1
        return None


@register_metric(PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME)
class WaitFirstReviewTimeCalculator(AverageMetricCalculator[timedelta]):
    """Elapsed time between requesting the review for the first time and getting it."""

    may_have_negative_values = False

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[timedelta]:
        """Calculate the actual state update."""
        if facts.first_review_request and facts.first_comment_on_first_review and \
                min_time <= facts.first_comment_on_first_review < max_time:
            return facts.first_comment_on_first_review - facts.first_review_request
        return None


@register_metric(PullRequestMetricID.PR_WAIT_FIRST_REVIEW_COUNT)
class WaitFirstReviewCounter(Counter):
    """Count PRs that were used to calculate PR_WAIT_FIRST_REVIEW_TIME disregarding \
    the quantiles."""

    deps = (WaitFirstReviewTimeCalculator,)


@register_metric(PullRequestMetricID.PR_WAIT_FIRST_REVIEW_COUNT_Q)
class WaitFirstReviewCounterWithQunatiles(CounterWithQuantiles):
    """Count PRs that were used to calculate PR_WAIT_FIRST_REVIEW_TIME respecting the quantiles."""

    deps = (WaitFirstReviewTimeCalculator,)


@register_metric(PullRequestMetricID.PR_OPENED)
class OpenedCalculator(SumMetricCalculator[int]):
    """Number of open PRs."""

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[int]:
        """Calculate the actual state update."""
        if min_time <= facts.created < max_time:
            return 1
        return None


@register_metric(PullRequestMetricID.PR_MERGED)
class MergedCalculator(SumMetricCalculator[int]):
    """Number of merged PRs."""

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[int]:
        """Calculate the actual state update."""
        if facts.merged and min_time <= facts.merged < max_time:
            return 1
        return None


@register_metric(PullRequestMetricID.PR_REJECTED)
class RejectedCalculator(SumMetricCalculator[int]):
    """Number of rejected PRs."""

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[int]:
        """Calculate the actual state update."""
        if facts.closed and not facts.merged and min_time <= facts.closed < max_time:
            return 1
        return None


@register_metric(PullRequestMetricID.PR_CLOSED)
class ClosedCalculator(SumMetricCalculator[int]):
    """Number of closed PRs."""

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[int]:
        """Calculate the actual state update."""
        if facts.closed and min_time <= facts.closed < max_time:
            return 1
        return None


@register_metric(PullRequestMetricID.PR_RELEASED)
class ReleasedCalculator(SumMetricCalculator[int]):
    """Number of released PRs."""

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[int]:
        """Calculate the actual state update."""
        if facts.released and min_time <= facts.released < max_time:
            return 1
        return None


@register_metric(PullRequestMetricID.PR_FLOW_RATIO)
class FlowRatioCalculator(MetricCalculator[float]):
    """PR flow ratio - opened / closed - calculator."""

    deps = (OpenedCalculator, ClosedCalculator)

    def __init__(self, *deps: MetricCalculator, quantiles: Sequence[float]):
        """Initialize a new instance of FlowRatioCalculator."""
        super().__init__(*deps, quantiles=quantiles)
        if isinstance(self._calcs[1], OpenedCalculator):
            self._calcs = list(reversed(self._calcs))
        self._opened, self._closed = self._calcs

    def _value(self, samples: Sequence[float]) -> Metric[float]:
        """Calculate the current metric value."""
        opened = self._opened.value
        closed = self._closed.value
        if not closed.exists and not opened.exists:
            return Metric(False, None, None, None)
        # Why +1? See ENG-866
        val = ((opened.value or 0) + 1) / ((closed.value or 0) + 1)
        return Metric(True, val, None, None)

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[float]:
        """Calculate the actual state update."""
        return None

    def _cut_by_quantiles(self) -> np.ndarray:
        return self._asarray()


@register_metric(PullRequestMetricID.PR_SIZE)
class SizeCalculator(AverageMetricCalculator[int]):
    """Average PR size.."""

    may_have_negative_values = False
    deps = (AllCounter,)

    def _shift_log(self, sample: int) -> int:
        return sample if sample > 0 else (sample + 1)

    def _analyze(self, facts: PullRequestFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[int]:
        """Calculate the actual state update."""
        if self._calcs[0].peek is not None:
            return facts.size
        return None
