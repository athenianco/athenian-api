from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence, Type

import numpy as np
import pandas as pd

from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    BinnedMetricCalculator, Counter, HistogramCalculator, HistogramCalculatorEnsemble, \
    MetricCalculator, MetricCalculatorEnsemble, SumMetricCalculator, WithoutQuantilesMixin
from athenian.api.models.web import PullRequestMetricID

metric_calculators: Dict[str, Type[MetricCalculator]] = {}
histogram_calculators: Dict[str, Type[HistogramCalculator]] = {}


class PullRequestMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of PullRequestMetricCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=metric_calculators)


class PullRequestHistogramCalculatorEnsemble(HistogramCalculatorEnsemble):
    """HistogramCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of PullRequestHistogramCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=histogram_calculators)


class PullRequestBinnedMetricCalculator(BinnedMetricCalculator):
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
    dtype = "timedelta64[s]"

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_time: datetime,
                 max_time: datetime,
                 override_event_time: Optional[datetime] = None,
                 override_event_indexes: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
        result = np.full(len(facts), None, object)
        wip_end = result.copy()
        no_last_review = facts["last_review"].isnull()
        has_last_review = ~no_last_review
        wip_end[has_last_review] = facts["first_review_request"].take(
            np.where(has_last_review)[0])

        # review was probably requested but never happened
        no_last_commit = facts["last_commit"].isnull()
        has_last_commit = ~no_last_commit & no_last_review
        wip_end[has_last_commit] = facts["last_commit"].take(np.where(has_last_commit)[0])

        # 0 commits in the PR, no reviews and review requests
        # => review time = 0
        # => merge time = 0 (you cannot merge an empty PR)
        # => release time = 0
        # This PR is 100% closed.
        remaining = np.where(wip_end == np.array(None))[0]
        closed = facts["closed"].take(remaining)
        wip_end[remaining] = closed
        wip_end[remaining[closed != closed]] = None  # deal with NaNs

        if override_event_time is not None:
            wip_end[override_event_indexes] = override_event_time

        wip_end_indexes = np.where(wip_end != np.array(None))[0]
        dtype = facts["created"].dtype
        wip_end = wip_end[wip_end_indexes].astype(dtype)
        min_time = np.array(min_time, dtype=dtype)
        max_time = np.array(max_time, dtype=dtype)
        wip_end_in_range = (min_time <= wip_end) & (wip_end < max_time)
        wip_end_indexes = wip_end_indexes[wip_end_in_range]
        result[wip_end_indexes] = (
            wip_end[wip_end_in_range] - facts["work_began"].take(wip_end_indexes).values
        ).astype(self.dtype).view(int)
        return result


@register_metric(PullRequestMetricID.PR_WIP_COUNT)
class WorkInProgressCounter(WithoutQuantilesMixin, Counter):
    """Count the number of PRs that were used to calculate PR_WIP_TIME \
    disregarding the quantiles."""

    deps = (WorkInProgressTimeCalculator,)


@register_metric(PullRequestMetricID.PR_WIP_COUNT_Q)
class WorkInProgressCounterWithQuantiles(Counter):
    """Count the number of PRs that were used to calculate PR_WIP_TIME respecting the quantiles."""

    deps = (WorkInProgressTimeCalculator,)


@register_metric(PullRequestMetricID.PR_REVIEW_TIME)
class ReviewTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time of the review process metric."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_time: datetime,
                 max_time: datetime,
                 allow_unclosed=False,
                 override_event_time: Optional[datetime] = None,
                 override_event_indexes: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
        result = np.full(len(facts), None, object)
        has_first_review_request = facts["first_review_request"].notnull()
        review_end = result.copy()
        # we cannot be sure that the approvals finished unless the PR is closed.
        if allow_unclosed:
            closed_mask = has_first_review_request
        else:
            closed_mask = facts["closed"].notnull() & has_first_review_request
        not_approved_mask = facts["approved"].isnull()
        approved_mask = ~not_approved_mask & closed_mask
        last_review_mask = not_approved_mask & facts["last_review"].notnull() & closed_mask
        review_end[approved_mask] = facts["approved"].take(np.where(approved_mask)[0])
        review_end[last_review_mask] = facts["last_review"].take(np.where(last_review_mask)[0])

        if override_event_time is not None:
            review_end[override_event_indexes] = override_event_time

        review_not_none = review_end != np.array(None)
        review_in_range = np.full(len(result), False)
        dtype = facts["created"].dtype
        review_end = review_end[review_not_none].astype(dtype)
        min_time = np.array(min_time, dtype=dtype)
        max_time = np.array(max_time, dtype=dtype)
        review_in_range_mask = (min_time <= review_end) & (review_end < max_time)
        review_in_range[review_not_none] = review_in_range_mask
        review_end = review_end[review_in_range_mask]
        if len(review_end):
            result[review_in_range] = (
                review_end -
                facts["first_review_request"].take(np.where(review_in_range)[0]).values
            ).astype(self.dtype).view(int)
        return result


@register_metric(PullRequestMetricID.PR_REVIEW_COUNT)
class ReviewCounter(WithoutQuantilesMixin, Counter):
    """Count the number of PRs that were used to calculate PR_REVIEW_TIME disregarding \
    the quantiles."""

    deps = (ReviewTimeCalculator,)


@register_metric(PullRequestMetricID.PR_REVIEW_COUNT_Q)
class ReviewCounterWithQuantiles(Counter):
    """Count the number of PRs that were used to calculate PR_REVIEW_TIME respecting \
    the quantiles."""

    deps = (ReviewTimeCalculator,)


@register_metric(PullRequestMetricID.PR_MERGING_TIME)
class MergingTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time to merge after finishing the review metric."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_time: datetime,
                 max_time: datetime,
                 override_event_time: Optional[datetime] = None,
                 override_event_indexes: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
        result = np.full(len(facts), None, object)
        merge_end = result.copy()
        closed_indexes = np.where(facts["closed"].notnull())[0]
        closed = facts["closed"].take(closed_indexes).values
        dtype = facts["created"].dtype
        min_time = np.array(min_time, dtype=dtype)
        max_time = np.array(max_time, dtype=dtype)
        closed_in_range = (min_time <= closed) & (closed < max_time)
        closed_indexes = closed_indexes[closed_in_range]
        merge_end[closed_indexes] = closed[closed_in_range]
        closed_mask = np.full(len(facts), False)
        closed_mask[closed_indexes] = True

        if override_event_time is not None:
            merge_end[override_event_indexes] = override_event_time
            closed_mask[override_event_indexes] = True

        dtype = facts["created"].dtype
        not_approved_mask = facts["approved"].isnull()
        approved_mask = ~not_approved_mask & closed_mask
        merge_end_approved = merge_end[approved_mask].astype(dtype)
        if len(merge_end_approved):
            result[approved_mask] = (
                merge_end_approved -
                facts["approved"].take(np.where(approved_mask)[0]).values
            ).astype(self.dtype).view(int)
        not_last_review_mask = facts["last_review"].isnull()
        last_review_mask = not_approved_mask & ~not_last_review_mask & closed_mask
        merge_end_last_reviewed = merge_end[last_review_mask].astype(dtype)
        if len(merge_end_last_reviewed):
            result[last_review_mask] = (
                merge_end_last_reviewed -
                facts["last_review"].take(np.where(last_review_mask)[0]).values
            ).astype(self.dtype).view(int)
        last_commit_mask = \
            not_approved_mask & not_last_review_mask & facts["last_commit"].notnull() & closed_mask
        merge_end_last_commit = merge_end[last_commit_mask].astype(dtype)
        if len(merge_end_last_commit):
            result[last_commit_mask] = (
                merge_end_last_commit -
                facts["last_commit"].take(np.where(last_commit_mask)[0]).values
            ).astype(self.dtype).view(int)
        return result


@register_metric(PullRequestMetricID.PR_MERGING_COUNT)
class MergingCounter(WithoutQuantilesMixin, Counter):
    """Count the number of PRs that were used to calculate PR_MERGING_TIME disregarding \
    the quantiles."""

    deps = (MergingTimeCalculator,)


@register_metric(PullRequestMetricID.PR_MERGING_COUNT_Q)
class MergingCounterWithQuantiles(Counter):
    """Count the number of PRs that were used to calculate PR_MERGING_TIME respecting \
    the quantiles."""

    deps = (MergingTimeCalculator,)


@register_metric(PullRequestMetricID.PR_RELEASE_TIME)
class ReleaseTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time to appear in a release after merging metric."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_time: datetime,
                 max_time: datetime,
                 override_event_time: Optional[datetime] = None,
                 override_event_indexes: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
        result = np.full(len(facts), None, object)
        release_end = result.copy()
        released_indexes = np.where(facts["released"].notnull())[0]
        released = facts["released"].take(released_indexes)
        released_in_range = (min_time <= released) & (released < max_time)
        released_indexes = released_indexes[released_in_range]
        release_end[released_indexes] = released.take(np.where(released_in_range)[0])
        released_mask = np.full(len(facts), False)
        released_mask[released_indexes] = True

        if override_event_time is not None:
            release_end[override_event_indexes] = override_event_time
            released_mask[override_event_indexes] = True

        result_mask = facts["merged"].notnull() & released_mask
        merged = facts["merged"].take(np.where(result_mask)[0]).values
        release_end = release_end[result_mask].astype(merged.dtype)
        if len(release_end):
            result[result_mask] = (release_end - merged).astype(self.dtype).view(int)
        return result


@register_metric(PullRequestMetricID.PR_RELEASE_COUNT)
class ReleaseCounter(WithoutQuantilesMixin, Counter):
    """Count the number of PRs that were used to calculate PR_RELEASE_TIME disregarding \
    the quantiles."""

    deps = (ReleaseTimeCalculator,)


@register_metric(PullRequestMetricID.PR_RELEASE_COUNT_Q)
class ReleaseCounterWithQuantiles(Counter):
    """Count the number of PRs that were used to calculate PR_RELEASE_TIME respecting \
    the quantiles."""

    deps = (ReleaseTimeCalculator,)


@register_metric(PullRequestMetricID.PR_LEAD_TIME)
class LeadTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time to appear in a release since starting working on the PR."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        result = np.full(len(facts), None, object)
        released_indexes = np.where(facts["released"].notnull())[0]
        released = facts["released"].take(released_indexes)
        released_in_range = (min_time <= released) & (released < max_time)
        released_indexes = released_indexes[released_in_range]
        released = released.take(np.where(released_in_range)[0]).values
        if len(released):
            result[released_indexes] = (
                released -
                facts["work_began"].take(released_indexes).values
            ).astype(self.dtype).view(int)
        return result


@register_metric(PullRequestMetricID.PR_LEAD_COUNT)
class LeadCounter(WithoutQuantilesMixin, Counter):
    """Count the number of PRs that were used to calculate PR_LEAD_TIME disregarding \
    the quantiles."""

    deps = (LeadTimeCalculator,)


@register_metric(PullRequestMetricID.PR_LEAD_COUNT_Q)
class LeadCounterWithQuantiles(Counter):
    """Count the number of PRs that were used to calculate PR_LEAD_TIME respecting \
    the quantiles."""

    deps = (LeadTimeCalculator,)


@register_metric(PullRequestMetricID.PR_CYCLE_TIME)
class CycleTimeCalculator(WithoutQuantilesMixin, MetricCalculator[timedelta]):
    """Sum of PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGE_TIME, and PR_RELEASE_TIME."""

    deps = (WorkInProgressTimeCalculator,
            ReviewTimeCalculator,
            MergingTimeCalculator,
            ReleaseTimeCalculator)
    dtype = "timedelta64[s]"

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

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        """Update the states of the underlying calcs and return whether at least one of the PR's \
        metrics exists."""
        sumval = np.full(len(facts), None, object)
        for calc in self._calcs:
            peek = calc.peek
            sum_none_mask = sumval == np.array(None)
            peek_not_none_mask = peek != np.array(None)
            copy_mask = sum_none_mask & peek_not_none_mask
            sumval[copy_mask] = peek[copy_mask]
            add_mask = ~sum_none_mask & peek_not_none_mask
            sumval[add_mask] += peek[add_mask]
        return sumval


@register_metric(PullRequestMetricID.PR_CYCLE_COUNT)
class CycleCounter(WithoutQuantilesMixin, Counter):
    """Count unique PRs that were used to calculate PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGE_TIME, \
    or PR_RELEASE_TIME disregarding the quantiles."""

    deps = (CycleTimeCalculator,)


@register_metric(PullRequestMetricID.PR_CYCLE_COUNT_Q)
class CycleCounterWithQuantiles(Counter):
    """Count unique PRs that were used to calculate PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGE_TIME, \
    or PR_RELEASE_TIME respecting the quantiles."""

    deps = (CycleTimeCalculator,)


@register_metric(PullRequestMetricID.PR_ALL_COUNT)
class AllCounter(SumMetricCalculator[int]):
    """Count all the PRs that are active in the given time interval."""

    dtype = int

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        cut_before_released = facts["released"] < min_time
        cut_before_rejected = (facts["closed"] < min_time) & facts["merged"].isnull()
        cut_after = facts["created"] >= max_time  # not `work_began`! Breaks granular measurements.
        cut_old_unreleased = (facts["merged"] < min_time) & facts["released"].isnull()
        # see also: ENG-673
        result = np.full(len(facts), None, object)
        result[~(cut_before_released | cut_before_rejected | cut_after | cut_old_unreleased)] = 1
        return result


@register_metric(PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME)
class WaitFirstReviewTimeCalculator(AverageMetricCalculator[timedelta]):
    """Elapsed time between requesting the review for the first time and getting it."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        result = np.full(len(facts), None, object)
        result_mask = facts["first_comment_on_first_review"].notnull() & \
            facts["first_review_request"].notnull()
        fc_on_fr = facts["first_comment_on_first_review"].take(np.where(result_mask)[0])
        fc_on_fr_in_range_mask = (min_time <= fc_on_fr) & (fc_on_fr < max_time)
        result_mask = np.where(result_mask)[0]
        result_mask = result_mask[fc_on_fr_in_range_mask]
        result[result_mask] = (
            fc_on_fr.take(np.where(fc_on_fr_in_range_mask)[0]).values -
            facts["first_review_request"].take(result_mask).values
        ).astype(self.dtype).view(int)
        return result


@register_metric(PullRequestMetricID.PR_WAIT_FIRST_REVIEW_COUNT)
class WaitFirstReviewCounter(WithoutQuantilesMixin, Counter):
    """Count PRs that were used to calculate PR_WAIT_FIRST_REVIEW_TIME disregarding \
    the quantiles."""

    deps = (WaitFirstReviewTimeCalculator,)


@register_metric(PullRequestMetricID.PR_WAIT_FIRST_REVIEW_COUNT_Q)
class WaitFirstReviewCounterWithQuantiles(Counter):
    """Count PRs that were used to calculate PR_WAIT_FIRST_REVIEW_TIME respecting the quantiles."""

    deps = (WaitFirstReviewTimeCalculator,)


@register_metric(PullRequestMetricID.PR_OPENED)
class OpenedCalculator(SumMetricCalculator[int]):
    """Number of open PRs."""

    dtype = int

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        created = facts["created"].values
        dtype = facts["created"].dtype
        min_time = np.array(min_time, dtype=dtype)
        max_time = np.array(max_time, dtype=dtype)
        result = np.full(len(facts), None, object)
        result[(min_time <= created) & (created < max_time)] = 1
        return result


@register_metric(PullRequestMetricID.PR_REVIEWED)
class ReviewedCalculator(SumMetricCalculator[int]):
    """Number of reviewed PRs."""

    dtype = int

    def _analyze(self, facts: np.ndarray, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        dtype = facts["created"].dtype
        min_time = np.array(min_time, dtype=dtype)
        max_time = np.array(max_time, dtype=dtype)
        review_timestamps = np.concatenate(facts["reviews"])
        reviews_in_range = (min_time <= review_timestamps) & (review_timestamps < max_time)
        # we cannot sum `reviews_in_range` because there can be several reviews for the same PR
        review_offsets = np.zeros(len(facts) + 1, dtype=int)
        np.cumsum(facts["reviews"].apply(len).values, out=review_offsets[1:])
        # np.searchsorted aliases several reviews of the same PR to the right border of a
        # `review_offsets` interval
        # np.unique collapses duplicate indexes
        reviewed_indexes = np.unique(np.searchsorted(
            review_offsets, np.where(reviews_in_range)[0], side="right") - 1)
        result = np.full(len(facts), None, object)
        result[reviewed_indexes] = 1
        return result


@register_metric(PullRequestMetricID.PR_MERGED)
class MergedCalculator(SumMetricCalculator[int]):
    """Number of merged PRs."""

    dtype = int

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        merged_indexes = np.where(facts["merged"].notnull())[0]
        merged = facts["merged"].take(merged_indexes).values
        dtype = facts["created"].dtype
        min_time = np.array(min_time, dtype=dtype)
        max_time = np.array(max_time, dtype=dtype)
        result = np.full(len(facts), None, object)
        result[merged_indexes[(min_time <= merged) & (merged < max_time)]] = 1
        return result


@register_metric(PullRequestMetricID.PR_REJECTED)
class RejectedCalculator(SumMetricCalculator[int]):
    """Number of rejected PRs."""

    dtype = int

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        rejected_indexes = np.where(facts["closed"].notnull() & facts["merged"].isnull())[0]
        closed = facts["closed"].take(rejected_indexes)
        dtype = facts["created"].dtype
        min_time = np.array(min_time, dtype=dtype)
        max_time = np.array(max_time, dtype=dtype)
        result = np.full(len(facts), None, object)
        result[rejected_indexes[(min_time <= closed) & (closed < max_time)]] = 1
        return result


@register_metric(PullRequestMetricID.PR_CLOSED)
class ClosedCalculator(SumMetricCalculator[int]):
    """Number of closed PRs."""

    dtype = int

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        closed_indexes = np.where(facts["closed"].notnull())[0]
        closed = facts["closed"].take(closed_indexes).values
        dtype = facts["created"].dtype
        min_time = np.array(min_time, dtype=dtype)
        max_time = np.array(max_time, dtype=dtype)
        result = np.full(len(facts), None, object)
        result[closed_indexes[(min_time <= closed) & (closed < max_time)]] = 1
        return result


@register_metric(PullRequestMetricID.PR_NOT_REVIEWED)
class NotReviewedCalculator(SumMetricCalculator[int]):
    """Number of non-reviewed PRs."""

    deps = (ReviewedCalculator, ClosedCalculator)
    dtype = int

    def _analyze(self, facts: np.ndarray, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        dtype = facts["created"].dtype
        min_time = np.array(min_time, dtype=dtype)
        max_time = np.array(max_time, dtype=dtype)
        not_reviewed = ~self._calcs[0].peek.astype(bool)
        closed = self._calcs[1].peek.astype(bool)
        result = np.full(len(facts), None, object)
        result[closed & not_reviewed] = 1
        return result


@register_metric(PullRequestMetricID.PR_DONE)
class ReleasedCalculator(SumMetricCalculator[int]):
    """Number of released PRs."""

    dtype = int

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        released_indexes = np.where(facts["released"].notnull())[0]
        released = facts["released"].take(released_indexes).values
        dtype = facts["created"].dtype
        min_time = np.array(min_time, dtype=dtype)
        max_time = np.array(max_time, dtype=dtype)
        result = np.full(len(facts), None, object)
        result[released_indexes[(min_time <= released) & (released < max_time)]] = 1

        rejected_indexes = np.where(
            facts["closed"].notnull().values
            &
            (facts["merged"].isnull().values | facts["force_push_dropped"]))[0]
        rejected = facts["closed"].take(rejected_indexes)
        result[rejected_indexes[(min_time <= rejected) & (rejected < max_time)]] = 1
        return result


@register_metric(PullRequestMetricID.PR_FLOW_RATIO)
class FlowRatioCalculator(WithoutQuantilesMixin, MetricCalculator[float]):
    """PR flow ratio - opened / closed - calculator."""

    deps = (OpenedCalculator, ClosedCalculator)
    dtype = float

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

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        return np.full(len(facts), None, object)


@register_metric(PullRequestMetricID.PR_SIZE)
class SizeCalculator(AverageMetricCalculator[int]):
    """Average PR size."""

    may_have_negative_values = False
    deps = (AllCounter,)
    dtype = int

    def _shift_log(self, sample: int) -> int:
        return sample if sample > 0 else (sample + 1)

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        sizes = facts["size"].values.astype(object)
        sizes[self._calcs[0].peek == np.array(None)] = None
        return sizes


class StagePendingDependencyCalculator(WithoutQuantilesMixin, SumMetricCalculator[int]):
    """Common dependency for stage-pending counters."""

    dtype = object

    def _analyze(self, facts: np.ndarray, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        merged_mask, approved_mask, frr_mask = (
            (v := facts[c].values) == v
            for c in ("merged", "approved", "first_review_request"))

        stage_indexes = {}
        other = ~facts["done"].values
        stage_indexes["release"] = np.where(merged_mask & other)[0]
        other &= ~merged_mask
        stage_indexes["merge"] = np.where(approved_mask & other)[0]
        other &= ~approved_mask
        stage_indexes["review"] = np.where(frr_mask & other)[0]
        other &= ~frr_mask
        stage_indexes["wip"] = np.where(other)[0]

        return np.array([stage_indexes], dtype=object)


class BaseStagePendingCounter(WithoutQuantilesMixin, SumMetricCalculator[int]):
    """Base stage-pending counter."""

    dtype = int
    stage_name = None
    deps = (StagePendingDependencyCalculator,)

    def _analyze(self, facts: np.ndarray, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        result = np.full(len(facts), None, object)
        result[self._calcs[0].peek[0][self.stage_name]] = 1
        return result


@register_metric(PullRequestMetricID.PR_WIP_PENDING_COUNT)
class WorkInProgressPendingCounter(BaseStagePendingCounter):
    """Number of PRs curently in wip stage."""

    stage_name = "wip"


@register_metric(PullRequestMetricID.PR_REVIEW_PENDING_COUNT)
class ReviewPendingCounter(BaseStagePendingCounter):
    """Number of PRs curently in review stage."""

    stage_name = "review"


@register_metric(PullRequestMetricID.PR_MERGING_PENDING_COUNT)
class MergingPendingCounter(BaseStagePendingCounter):
    """Number of PRs curently in merge stage."""

    stage_name = "merge"


@register_metric(PullRequestMetricID.PR_RELEASE_PENDING_COUNT)
class ReleasePendingCounter(BaseStagePendingCounter):
    """Number of PRs curently in release stage."""

    stage_name = "release"
