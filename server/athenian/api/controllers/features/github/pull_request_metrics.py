from datetime import datetime, timedelta
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type

import numpy as np
import pandas as pd

from athenian.api.controllers.features.histogram import Histogram
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    BinnedEnsemblesCalculator, BinnedMetricsCalculator, Counter, HistogramCalculator, \
    HistogramCalculatorEnsemble, \
    MetricCalculator, MetricCalculatorEnsemble, SumMetricCalculator, WithoutQuantilesMixin
from athenian.api.models.web import PullRequestMetricID

metric_calculators: Dict[str, Type[MetricCalculator]] = {}
histogram_calculators: Dict[str, Type[HistogramCalculator]] = {}


class PullRequestMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float], exclude_inactive: bool = False):
        """Initialize a new instance of PullRequestMetricCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=metric_calculators)
        for calc in self._calcs:
            calc.exclude_inactive = exclude_inactive


class PullRequestHistogramCalculatorEnsemble(HistogramCalculatorEnsemble):
    """HistogramCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of PullRequestHistogramCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=histogram_calculators)


class PullRequestBinnedMetricCalculator(BinnedMetricsCalculator):
    """BinnedMetricCalculator adapted for pull requests."""

    ensemble_class = PullRequestMetricCalculatorEnsemble


class PullRequestBinnedHistogramCalculator(BinnedEnsemblesCalculator[Histogram]):
    """BinnedEnsemblesCalculator adapted for pull request histograms."""

    ensemble_class = PullRequestHistogramCalculatorEnsemble

    def _aggregate_ensembles(self, kwargs: Iterable[Dict[str, Any]],
                             ) -> List[Dict[str, List[List[Metric]]]]:
        return [{k: v for k, v in ensemble.histograms(**ekw).items()}
                for ensemble, ekw in zip(self.ensembles, kwargs)]


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
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 override_event_time: Optional[datetime] = None,
                 override_event_indexes: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
        wip_end = np.full(len(facts), None, object)
        no_last_review = facts["last_review"].isnull().values
        has_last_review = ~no_last_review
        wip_end[has_last_review] = facts["first_review_request"].take(
            np.where(has_last_review)[0])

        # review was probably requested but never happened
        no_last_commit = facts["last_commit"].isnull().values
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
        wip_end_in_range = (min_times[:, None] <= wip_end) & (wip_end < max_times[:, None])
        result = np.full((len(min_times), len(facts)), None, object)
        work_began = facts["work_began"].values
        for result_dim, wip_end_in_range_dim in zip(result, wip_end_in_range):
            wip_end_indexes_dim = wip_end_indexes[wip_end_in_range_dim]
            result_dim[wip_end_indexes_dim] = (
                wip_end[wip_end_in_range_dim] - work_began[wip_end_indexes_dim]
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
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 allow_unclosed=False,
                 override_event_time: Optional[datetime] = None,
                 override_event_indexes: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
        result = np.full(len(facts), None, object)
        has_first_review_request = facts["first_review_request"].notnull().values
        review_end = result.copy()
        # we cannot be sure that the approvals finished unless the PR is closed.
        if allow_unclosed:
            closed_mask = has_first_review_request
        else:
            closed_mask = facts["closed"].notnull().values & has_first_review_request
        not_approved_mask = facts["approved"].isnull()
        approved_mask = ~not_approved_mask & closed_mask
        last_review_mask = not_approved_mask & facts["last_review"].notnull() & closed_mask
        review_end[approved_mask] = facts["approved"].take(np.where(approved_mask)[0])
        review_end[last_review_mask] = facts["last_review"].take(np.where(last_review_mask)[0])

        if override_event_time is not None:
            review_end[override_event_indexes] = override_event_time

        review_not_none = review_end != np.array(None)
        review_in_range = np.full((len(min_times), len(result)), False)
        dtype = facts["created"].dtype
        review_end = review_end[review_not_none].astype(dtype)
        review_in_range_mask = \
            (min_times[:, None] <= review_end) & (review_end < max_times[:, None])
        review_in_range[:, review_not_none] = review_in_range_mask
        review_end = np.repeat(review_end[None, :], len(min_times), axis=0)
        result = np.repeat(result[None, :], len(min_times), axis=0)
        frr = facts["first_review_request"].values
        for result_dim, review_end_dim, review_in_range_mask_dim, review_in_range_dim in \
                zip(result, review_end, review_in_range_mask, review_in_range):
            review_end_dim = review_end_dim[review_in_range_mask_dim]
            result_dim[review_in_range_dim] = (
                review_end_dim - frr[review_in_range_dim]
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
    """Time to close PR after finishing the review metric."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 override_event_time: Optional[datetime] = None,
                 override_event_indexes: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        merge_end = result.copy()
        closed_indexes = np.where(facts["closed"].notnull())[0]
        closed = facts["closed"].take(closed_indexes).values
        closed_in_range = (min_times[:, None] <= closed) & (closed < max_times[:, None])
        closed_indexes = np.repeat(closed_indexes[None, :], len(min_times), axis=0)
        closed_mask = np.full((len(min_times), len(facts)), False)
        for merge_end_dim, closed_mask_dim, closed_in_range_dim, closed_indexes_dim in \
                zip(merge_end, closed_mask, closed_in_range, closed_indexes):
            closed_indexes_dim = closed_indexes_dim[closed_in_range_dim]
            merge_end_dim[closed_indexes_dim] = closed[closed_in_range_dim]
            closed_mask_dim[closed_indexes_dim] = True

        if override_event_time is not None:
            merge_end[:, override_event_indexes] = override_event_time
            closed_mask[:, override_event_indexes] = True

        dtype = facts["created"].dtype
        not_approved_mask = \
            np.repeat(facts["approved"].isnull().values[None, :], len(min_times), axis=0)
        approved_mask = ~not_approved_mask & closed_mask
        merge_end_approved = merge_end[approved_mask].astype(dtype)
        approved = np.repeat(facts["approved"].values[None, :], len(min_times), axis=0)
        result[approved_mask] = (
            merge_end_approved - approved[approved_mask]
        ).astype(self.dtype).view(int)
        not_last_review_mask = \
            np.repeat(facts["last_review"].isnull().values[None, :], len(min_times), axis=0)
        last_review_mask = ~not_last_review_mask & not_approved_mask & closed_mask
        merge_end_last_reviewed = merge_end[last_review_mask].astype(dtype)
        last_review = np.repeat(facts["last_review"].values[None, :], len(min_times), axis=0)
        result[last_review_mask] = (
            merge_end_last_reviewed - last_review[last_review_mask]
        ).astype(self.dtype).view(int)
        has_last_commit = \
            np.repeat(facts["last_commit"].notnull().values[None, :], len(min_times), axis=0)
        last_commit_mask = \
            not_approved_mask & not_last_review_mask & has_last_commit & closed_mask
        merge_end_last_commit = merge_end[last_commit_mask].astype(dtype)
        last_commit = np.repeat(facts["last_commit"].values[None, :], len(min_times), axis=0)
        result[last_commit_mask] = (
            merge_end_last_commit - last_commit[last_commit_mask]
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
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 override_event_time: Optional[datetime] = None,
                 override_event_indexes: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        released = facts["released"].values
        released_mask = (min_times[:, None] <= released) & (released < max_times[:, None])
        release_end = result.copy()
        release_end[released_mask] = \
            np.repeat(released[None, :], len(min_times), axis=0)[released_mask]

        if override_event_time is not None:
            release_end[:, override_event_indexes] = override_event_time
            released_mask[:, override_event_indexes] = True

        result_mask = released_mask
        result_mask[:, facts["merged"].isnull().values] = False
        merged = np.repeat(facts["merged"].values[None, :], len(min_times), axis=0)[result_mask]
        release_end = release_end[result_mask].astype(merged.dtype)
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

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        released_indexes = np.where(facts["released"].notnull())[0]
        released = facts["released"].take(released_indexes).values
        released_in_range = (min_times[:, None] <= released) & (released < max_times[:, None])
        work_began = facts["work_began"].values
        for result_dim, released_in_range_dim in zip(result, released_in_range):
            released_indexes_dim = released_indexes[released_in_range_dim]
            result_dim[released_indexes_dim] = (
                released[released_in_range_dim] - work_began[released_indexes_dim]
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

    def _values(self) -> List[List[Metric[timedelta]]]:
        """Calculate the current metric value."""
        def zeros(v):
            return [[v for _ in range(len(gs))] for gs in self.samples]

        ct = zeros(timedelta(0))
        ct_conf_min = zeros(timedelta(0))
        ct_conf_max = zeros(timedelta(0))
        exists = zeros(False)
        for calc in self._calcs:
            values = calc.values
            for i, gm in enumerate(values):
                for j, m in enumerate(gm):
                    if m.exists:
                        exists[i][j] = True
                        ct[i][j] += m.value
                        ct_conf_min[i][j] += m.confidence_min
                        ct_conf_max[i][j] += m.confidence_max
        metrics = [
            [Metric(texists, tct if texists else None,
                    tct_conf_min if texists else None, tct_conf_max if texists else None)
             for texists, tct, tct_conf_min, tct_conf_max in zip(
                gexists, gct, gct_conf_min, gct_conf_max)]
            for gexists, gct, gct_conf_min, gct_conf_max in zip(
                exists, ct, ct_conf_min, ct_conf_max)
        ]
        return metrics

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        """Update the states of the underlying calcs and return whether at least one of the PR's \
        metrics exists."""
        sumval = np.full((len(min_times), len(facts)), None, object)
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
    exclude_inactive = False

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        created_in_range_mask = facts["created"].values < max_times[:, None]
        released = facts["released"].values
        released_in_range_mask = released >= min_times[:, None]
        closed = facts["closed"].values
        closed_in_range_mask = (closed >= min_times[:, None]) | (closed != closed)
        if not self.exclude_inactive:
            merged_unreleased_mask = \
                (facts["merged"].values < min_times[:, None]) & (released != released)
        else:
            merged_unreleased_mask = np.array([False])
            # we should intersect each PR's activity days with [min_times, max_times).
            # the following is similar to ReviewedCalculator
            activity_mask = np.full((len(min_times), len(facts)), False)
            activity_days = np.concatenate(facts["activity_days"]).astype(facts["created"].dtype)
            activities_in_range = (
                (min_times[:, None] <= activity_days)
                &
                (activity_days < max_times[:, None])
            )
            activity_offsets = np.zeros(len(facts) + 1, dtype=int)
            np.cumsum(facts["activity_days"].apply(len).values, out=activity_offsets[1:])
            for activity_mask_dim, activities_in_range_dim in \
                    zip(activity_mask, activities_in_range):
                activity_indexes = np.unique(np.searchsorted(
                    activity_offsets, np.where(activities_in_range_dim)[0], side="right") - 1)
                activity_mask_dim[activity_indexes] = 1

        in_range_mask = created_in_range_mask & (
            released_in_range_mask | closed_in_range_mask | merged_unreleased_mask
        )
        if self.exclude_inactive:
            in_range_mask &= activity_mask
        result = np.full((len(min_times), len(facts)), None, object)
        result[in_range_mask] = 1
        return result


@register_metric(PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME)
class WaitFirstReviewTimeCalculator(AverageMetricCalculator[timedelta]):
    """Elapsed time between requesting the review for the first time and getting it."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        result_mask = facts["first_comment_on_first_review"].notnull().values & \
            facts["first_review_request"].notnull().values
        fc_on_fr = facts["first_comment_on_first_review"].take(np.where(result_mask)[0]).values
        fc_on_fr_in_range_mask = (min_times[:, None] <= fc_on_fr) & (fc_on_fr < max_times[:, None])
        result_indexes = np.where(result_mask)[0]
        first_review_request = facts["first_review_request"].values
        for result_dim, fc_on_fr_in_range_mask_dim in zip(result, fc_on_fr_in_range_mask):
            result_indexes_dim = result_indexes[fc_on_fr_in_range_mask_dim]
            result_dim[result_indexes_dim] = (
                fc_on_fr[fc_on_fr_in_range_mask_dim] - first_review_request[result_indexes_dim]
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

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        created = facts["created"].values
        result = np.full((len(min_times), len(facts)), None, object)
        result[(min_times[:, None] <= created) & (created < max_times[:, None])] = 1
        return result


@register_metric(PullRequestMetricID.PR_REVIEWED)
class ReviewedCalculator(SumMetricCalculator[int]):
    """Number of reviewed PRs."""

    dtype = int

    def _analyze(self,
                 facts: np.ndarray,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        review_timestamps = np.concatenate(facts["reviews"]).astype(facts["created"].dtype)
        reviews_in_range = \
            (min_times[:, None] <= review_timestamps) & (review_timestamps < max_times[:, None])
        # we cannot sum `reviews_in_range` because there can be several reviews for the same PR
        review_offsets = np.zeros(len(facts) + 1, dtype=int)
        np.cumsum(facts["reviews"].apply(len).values, out=review_offsets[1:])
        result = np.full((len(min_times), len(facts)), None, object)
        for result_dim, reviews_in_range_dim in zip(result, reviews_in_range):
            # np.searchsorted aliases several reviews of the same PR to the right border of a
            # `review_offsets` interval
            # np.unique collapses duplicate indexes
            reviewed_indexes = np.unique(np.searchsorted(
                review_offsets, np.where(reviews_in_range_dim)[0], side="right") - 1)
            result_dim[reviewed_indexes] = 1
        return result


@register_metric(PullRequestMetricID.PR_MERGED)
class MergedCalculator(SumMetricCalculator[int]):
    """Number of merged PRs."""

    dtype = int

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        merged = facts["merged"].values
        result = np.full((len(min_times), len(facts)), None, object)
        result[(min_times[:, None] <= merged) & (merged < max_times[:, None])] = 1
        return result


@register_metric(PullRequestMetricID.PR_REJECTED)
class RejectedCalculator(SumMetricCalculator[int]):
    """Number of rejected PRs."""

    dtype = int

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        closed = facts["closed"].values
        closed_in_range_mask = (min_times[:, None] <= closed) & (closed < max_times[:, None])
        unmerged_mask = facts["merged"].isnull().values
        result = np.full((len(min_times), len(facts)), None, object)
        result[closed_in_range_mask & unmerged_mask] = 1
        return result


@register_metric(PullRequestMetricID.PR_CLOSED)
class ClosedCalculator(SumMetricCalculator[int]):
    """Number of closed PRs."""

    dtype = int

    def _analyze(self, facts: pd.DataFrame, min_times: np.ndarray, max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        closed = facts["closed"].values
        result = np.full((len(min_times), len(facts)), None, object)
        result[(min_times[:, None] <= closed) & (closed < max_times[:, None])] = 1
        return result


@register_metric(PullRequestMetricID.PR_NOT_REVIEWED)
class NotReviewedCalculator(SumMetricCalculator[int]):
    """Number of non-reviewed PRs."""

    deps = (ReviewedCalculator, ClosedCalculator)
    dtype = int

    def _analyze(self,
                 facts: np.ndarray,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        not_reviewed = ~self._calcs[0].peek.astype(bool)
        closed = self._calcs[1].peek.astype(bool)
        result = np.full((len(min_times), len(facts)), None, object)
        result[closed & not_reviewed] = 1
        return result


@register_metric(PullRequestMetricID.PR_DONE)
class ReleasedCalculator(SumMetricCalculator[int]):
    """Number of released PRs."""

    dtype = int

    def _analyze(self, facts: pd.DataFrame, min_times: np.ndarray, max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        released = facts["released"].values
        result = np.full((len(min_times), len(facts)), None, object)
        result[(min_times[:, None] <= released) & (released < max_times[:, None])] = 1
        rejected_mask = (
            facts["closed"].notnull().values
            &
            (facts["merged"].isnull().values | facts["force_push_dropped"].values)
        )
        closed = facts["closed"].values
        result[(min_times[:, None] <= closed) & (closed < max_times[:, None]) & rejected_mask] = 1
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

    def _values(self) -> List[List[Metric[float]]]:
        """Calculate the current metric value."""
        metrics = [[Metric(False, None, None, None)] * len(samples) for samples in self.samples]
        for i, (opened_group, closed_group) in enumerate(zip(
                self._opened.values, self._closed.values)):
            for j, (opened, closed) in enumerate(zip(opened_group, closed_group)):
                if not closed.exists and not opened.exists:
                    continue
                # Why +1? See ENG-866
                val = ((opened.value or 0) + 1) / ((closed.value or 0) + 1)
                metrics[i][j] = Metric(True, val, None, None)
        return metrics

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        return np.full((len(min_times), len(facts)), None, object)


@register_metric(PullRequestMetricID.PR_SIZE)
class SizeCalculator(AverageMetricCalculator[int]):
    """Average PR size."""

    may_have_negative_values = False
    deps = (AllCounter,)
    dtype = int

    def _shift_log(self, samples: np.ndarray) -> np.ndarray:
        samples[samples == 0] = 1
        return samples

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        sizes = np.repeat(facts["size"].values[None, :], len(min_times), axis=0).astype(object)
        sizes[self._calcs[0].peek == np.array(None)] = None
        return sizes


class PendingStage(IntEnum):
    """Indexes of the pending stages that are used below."""

    WIP = 0
    REVIEW = 1
    MERGE = 2
    RELEASE = 3


class StagePendingDependencyCalculator(WithoutQuantilesMixin, SumMetricCalculator[int]):
    """Common dependency for stage-pending counters."""

    dtype = int
    deps = (AllCounter,)

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        merged_mask, approved_mask, frr_mask = (
            facts[c].notnull().values
            for c in ("merged", "approved", "first_review_request"))

        stage_masks = np.full((len(min_times), len(facts), len(PendingStage)), None, object)
        other = ~facts["done"].values
        stage_masks[:, merged_mask & other, PendingStage.RELEASE] = True
        other &= ~merged_mask
        stage_masks[:, approved_mask & other, PendingStage.MERGE] = True
        other &= ~approved_mask
        stage_masks[:, frr_mask & other, PendingStage.REVIEW] = True
        other &= ~frr_mask
        stage_masks[:, other, PendingStage.WIP] = True
        stage_masks[self._calcs[0].peek == np.array(None)] = None
        return stage_masks


class BaseStagePendingCounter(WithoutQuantilesMixin, SumMetricCalculator[int]):
    """Base stage-pending counter."""

    dtype = int
    stage = None
    deps = (StagePendingDependencyCalculator,)

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        return self._calcs[0].peek[:, :, self.stage]


@register_metric(PullRequestMetricID.PR_WIP_PENDING_COUNT)
class WorkInProgressPendingCounter(BaseStagePendingCounter):
    """Number of PRs currently in wip stage."""

    stage = PendingStage.WIP


@register_metric(PullRequestMetricID.PR_REVIEW_PENDING_COUNT)
class ReviewPendingCounter(BaseStagePendingCounter):
    """Number of PRs currently in review stage."""

    stage = PendingStage.REVIEW


@register_metric(PullRequestMetricID.PR_MERGING_PENDING_COUNT)
class MergingPendingCounter(BaseStagePendingCounter):
    """Number of PRs currently in merge stage."""

    stage = PendingStage.MERGE


@register_metric(PullRequestMetricID.PR_RELEASE_PENDING_COUNT)
class ReleasePendingCounter(BaseStagePendingCounter):
    """Number of PRs currently in release stage."""

    stage = PendingStage.RELEASE
