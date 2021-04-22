from datetime import datetime, timedelta
from enum import IntEnum
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Type

import numpy as np
import pandas as pd

from athenian.api import metadata
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    BinnedHistogramCalculator, BinnedMetricCalculator, Counter, HistogramCalculator, \
    HistogramCalculatorEnsemble, make_register_metric, MetricCalculator, \
    MetricCalculatorEnsemble, RatioCalculator, SumMetricCalculator, WithoutQuantilesMixin
from athenian.api.controllers.miners.types import PRParticipants, PullRequestFacts
from athenian.api.models.web import PullRequestMetricID


metric_calculators: Dict[str, Type[MetricCalculator]] = {}
histogram_calculators: Dict[str, Type[HistogramCalculator]] = {}
register_metric = make_register_metric(metric_calculators, histogram_calculators)


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


def group_by_lines(lines: Sequence[int], items: pd.DataFrame) -> List[np.ndarray]:
    """
    Bin PRs by number of changed `lines`.

    We throw away the ends: PRs with fewer lines than `lines[0]` and with more lines than \
    `lines[-1]`.

    :param lines: Either an empty sequence or one with at least 2 elements. The numbers must \
                  monotonically increase.
    """
    lines = np.asarray(lines)
    if len(lines) and not items.empty:
        assert len(lines) >= 2
        assert (np.diff(lines) > 0).all()
    else:
        return [np.arange(len(items))]
    values = items["size"].values
    line_group_assignments = np.digitize(values, lines)
    line_group_assignments[line_group_assignments == len(lines)] = 0
    line_group_assignments -= 1
    order = np.argsort(line_group_assignments)
    existing_groups, existing_group_counts = np.unique(
        line_group_assignments[order], return_counts=True)
    line_groups = np.split(np.arange(len(items))[order], np.cumsum(existing_group_counts)[:-1])
    if line_group_assignments[order[0]] < 0:
        line_groups = line_groups[1:]
        existing_groups = existing_groups[1:]
    full_line_groups = [np.array([], dtype=int)] * (len(lines) - 1)
    for i, g in zip(existing_groups, line_groups):
        full_line_groups[i] = g
    return full_line_groups


def group_prs_by_participants(participants: List[PRParticipants],
                              items: pd.DataFrame,
                              ) -> List[np.ndarray]:
    """
    Group PRs by participants.

    The aggregation is OR. We don't support all kinds, see `PullRequestFacts`'s mutable fields.
    """
    # if len(participants) == 1, we've already filtered in SQL so don't have to re-check
    if len(participants) < 2 or items.empty:
        return [np.arange(len(items))]
    groups = []
    log = logging.getLogger("%s.group_prs_by_participants" % metadata.__package__)
    for participants_group in participants:
        group = np.full(len(items), False)
        for participation_kind, devs in participants_group.items():
            name = participation_kind.name.lower()
            try:
                group |= items[name].isin(devs).values
            except KeyError:
                log.warning("Unsupported participation kind: %s", name)
        groups.append(np.nonzero(group)[0])
    return groups


class PullRequestBinnedMetricCalculator(BinnedMetricCalculator):
    """BinnedMetricCalculator adapted for pull requests."""

    ensemble_class = PullRequestMetricCalculatorEnsemble


class PullRequestBinnedHistogramCalculator(BinnedHistogramCalculator):
    """BinnedHistogramCalculator adapted for pull request histograms."""

    ensemble_class = PullRequestHistogramCalculatorEnsemble


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
        no_last_review = facts[PullRequestFacts.f.last_review].isnull().values
        has_last_review = ~no_last_review
        wip_end[has_last_review] = facts[PullRequestFacts.f.first_review_request].take(
            np.nonzero(has_last_review)[0])

        # review was probably requested but never happened
        no_last_commit = facts[PullRequestFacts.f.last_commit].isnull().values
        has_last_commit = ~no_last_commit & no_last_review
        wip_end[has_last_commit] = facts[PullRequestFacts.f.last_commit].take(np.nonzero(
            has_last_commit)[0])

        # 0 commits in the PR, no reviews and review requests
        # => review time = 0
        # => merge time = 0 (you cannot merge an empty PR)
        # => release time = 0
        # This PR is 100% closed.
        remaining = np.nonzero(wip_end == np.array(None))[0]
        closed = facts[PullRequestFacts.f.closed].take(remaining)
        wip_end[remaining] = closed
        wip_end[remaining[closed != closed]] = None  # deal with NaNs

        if override_event_time is not None:
            wip_end[override_event_indexes] = override_event_time

        wip_end_indexes = np.nonzero(wip_end != np.array(None))[0]
        dtype = facts[PullRequestFacts.f.created].dtype
        wip_end = wip_end[wip_end_indexes].astype(dtype)
        wip_end_in_range = (min_times[:, None] <= wip_end) & (wip_end < max_times[:, None])
        result = np.full((len(min_times), len(facts)), None, object)
        work_began = facts[PullRequestFacts.f.work_began].values
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
        has_first_review_request = \
            facts[PullRequestFacts.f.first_review_request].notnull().values
        review_end = result.copy()
        # we cannot be sure that the approvals finished unless the PR is closed.
        if allow_unclosed:
            closed_mask = has_first_review_request
        else:
            closed_mask = (facts[PullRequestFacts.f.closed].notnull().values &
                           has_first_review_request)
        not_approved_mask = facts[PullRequestFacts.f.approved].isnull().values
        approved_mask = ~not_approved_mask & closed_mask
        last_review_mask = (not_approved_mask &
                            facts[PullRequestFacts.f.last_review].notnull().values &
                            closed_mask)
        review_end[approved_mask] = facts[PullRequestFacts.f.approved].take(
            np.nonzero(approved_mask)[0])
        review_end[last_review_mask] = facts[PullRequestFacts.f.last_review].take(
            np.nonzero(last_review_mask)[0])

        if override_event_time is not None:
            review_end[override_event_indexes] = override_event_time

        review_not_none = review_end != np.array(None)
        review_in_range = np.full((len(min_times), len(result)), False)
        dtype = facts[PullRequestFacts.f.created].dtype
        review_end = review_end[review_not_none].astype(dtype)
        review_in_range_mask = \
            (min_times[:, None] <= review_end) & (review_end < max_times[:, None])
        review_in_range[:, review_not_none] = review_in_range_mask
        review_end = np.repeat(review_end[None, :], len(min_times), axis=0)
        result = np.repeat(result[None, :], len(min_times), axis=0)
        frr = facts[PullRequestFacts.f.first_review_request].values
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
        closed_indexes = np.nonzero(facts[PullRequestFacts.f.closed].notnull().values)[0]
        closed = facts[PullRequestFacts.f.closed].take(closed_indexes).values
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

        dtype = facts[PullRequestFacts.f.created].dtype
        not_approved_mask = \
            np.repeat(facts[PullRequestFacts.f.approved].isnull().values[None, :],
                      len(min_times), axis=0)
        approved_mask = ~not_approved_mask & closed_mask
        merge_end_approved = merge_end[approved_mask].astype(dtype)
        approved = np.repeat(facts[PullRequestFacts.f.approved].values[None, :],
                             len(min_times), axis=0)
        result[approved_mask] = (
            merge_end_approved - approved[approved_mask]
        ).astype(self.dtype).view(int)
        not_last_review_mask = \
            np.repeat(facts[PullRequestFacts.f.last_review].isnull().values[None, :],
                      len(min_times), axis=0)
        last_review_mask = ~not_last_review_mask & not_approved_mask & closed_mask
        merge_end_last_reviewed = merge_end[last_review_mask].astype(dtype)
        last_review = np.repeat(facts[PullRequestFacts.f.last_review].values[None, :],
                                len(min_times), axis=0)
        result[last_review_mask] = (
            merge_end_last_reviewed - last_review[last_review_mask]
        ).astype(self.dtype).view(int)
        has_last_commit = \
            np.repeat(facts[PullRequestFacts.f.last_commit].notnull().values[None, :],
                      len(min_times), axis=0)
        last_commit_mask = \
            not_approved_mask & not_last_review_mask & has_last_commit & closed_mask
        merge_end_last_commit = merge_end[last_commit_mask].astype(dtype)
        last_commit = np.repeat(facts[PullRequestFacts.f.last_commit].values[None, :],
                                len(min_times), axis=0)
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
        released = facts[PullRequestFacts.f.released].values
        released_mask = (min_times[:, None] <= released) & (released < max_times[:, None])
        release_end = result.copy()
        release_end[released_mask] = \
            np.repeat(released[None, :], len(min_times), axis=0)[released_mask]

        if override_event_time is not None:
            release_end[:, override_event_indexes] = override_event_time
            released_mask[:, override_event_indexes] = True

        result_mask = released_mask
        result_mask[:, facts[PullRequestFacts.f.merged].isnull().values] = False
        merged = np.repeat(facts[PullRequestFacts.f.merged].values[None, :],
                           len(min_times), axis=0)[result_mask]
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
        released_indexes = \
            np.nonzero(facts[PullRequestFacts.f.released].notnull().values)[0]
        released = facts[PullRequestFacts.f.released].take(released_indexes).values
        released_in_range = (min_times[:, None] <= released) & (released < max_times[:, None])
        work_began = facts[PullRequestFacts.f.work_began].values
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
        created_in_range_mask = \
            facts[PullRequestFacts.f.created].values < max_times[:, None]
        released = facts[PullRequestFacts.f.released].values
        released_in_range_mask = released >= min_times[:, None]
        closed = facts[PullRequestFacts.f.closed].values
        closed_in_range_mask = (closed >= min_times[:, None]) | (closed != closed)
        if not self.exclude_inactive:
            merged_unreleased_mask = (
                (facts[PullRequestFacts.f.merged].values < min_times[:, None]) &
                (released != released)
            )
        else:
            merged_unreleased_mask = np.array([False])
            # we should intersect each PR's activity days with [min_times, max_times).
            # the following is similar to ReviewedCalculator
            activity_mask = np.full((len(min_times), len(facts)), False)
            activity_days = np.concatenate(facts[PullRequestFacts.f.activity_days]) \
                .astype(facts[PullRequestFacts.f.created].dtype)
            activities_in_range = (
                (min_times[:, None] <= activity_days)
                &
                (activity_days < max_times[:, None])
            )
            activity_offsets = np.zeros(len(facts) + 1, dtype=int)
            np.cumsum(facts[PullRequestFacts.f.activity_days].apply(len).values,
                      out=activity_offsets[1:])
            for activity_mask_dim, activities_in_range_dim in \
                    zip(activity_mask, activities_in_range):
                activity_indexes = np.unique(np.searchsorted(
                    activity_offsets, np.nonzero(activities_in_range_dim)[0], side="right") - 1)
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
        result_mask = (
            facts[PullRequestFacts.f.first_comment_on_first_review].notnull().values &
            facts[PullRequestFacts.f.first_review_request].notnull().values
        )
        fc_on_fr = facts[PullRequestFacts.f.first_comment_on_first_review].take(
            np.nonzero(result_mask)[0]).values
        fc_on_fr_in_range_mask = (min_times[:, None] <= fc_on_fr) & (fc_on_fr < max_times[:, None])
        result_indexes = np.nonzero(result_mask)[0]
        first_review_request = facts[PullRequestFacts.f.first_review_request].values
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
        created = facts[PullRequestFacts.f.created].values
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
        review_timestamps = np.concatenate(facts[PullRequestFacts.f.reviews]) \
            .astype(facts[PullRequestFacts.f.created].dtype)
        reviews_in_range = \
            (min_times[:, None] <= review_timestamps) & (review_timestamps < max_times[:, None])
        # we cannot sum `reviews_in_range` because there can be several reviews for the same PR
        review_offsets = np.zeros(len(facts) + 1, dtype=int)
        np.cumsum(facts[PullRequestFacts.f.reviews].apply(len).values,
                  out=review_offsets[1:])
        result = np.full((len(min_times), len(facts)), None, object)
        for result_dim, reviews_in_range_dim in zip(result, reviews_in_range):
            # np.searchsorted aliases several reviews of the same PR to the right border of a
            # `review_offsets` interval
            # np.unique collapses duplicate indexes
            reviewed_indexes = np.unique(np.searchsorted(
                review_offsets, np.nonzero(reviews_in_range_dim)[0], side="right") - 1)
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
        merged = facts[PullRequestFacts.f.merged].values
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
        closed = facts[PullRequestFacts.f.closed].values
        closed_in_range_mask = (min_times[:, None] <= closed) & (closed < max_times[:, None])
        unmerged_mask = facts[PullRequestFacts.f.merged].isnull().values
        result = np.full((len(min_times), len(facts)), None, object)
        result[closed_in_range_mask & unmerged_mask] = 1
        return result


@register_metric(PullRequestMetricID.PR_CLOSED)
class ClosedCalculator(SumMetricCalculator[int]):
    """Number of closed PRs."""

    dtype = int

    def _analyze(self, facts: pd.DataFrame, min_times: np.ndarray, max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        closed = facts[PullRequestFacts.f.closed].values
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
class DoneCalculator(SumMetricCalculator[int]):
    """Number of rejected or released PRs."""

    dtype = int

    def _analyze(self, facts: pd.DataFrame, min_times: np.ndarray, max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        released = facts[PullRequestFacts.f.released].values
        result = np.full((len(min_times), len(facts)), None, object)
        result[(min_times[:, None] <= released) & (released < max_times[:, None])] = 1
        rejected_mask = (
            facts[PullRequestFacts.f.closed].notnull().values
            &
            (facts[PullRequestFacts.f.merged].isnull().values |
             facts[PullRequestFacts.f.force_push_dropped].values)
        )
        closed = facts[PullRequestFacts.f.closed].values
        result[(min_times[:, None] <= closed) & (closed < max_times[:, None]) & rejected_mask] = 1
        return result


@register_metric(PullRequestMetricID.PR_FLOW_RATIO)
class FlowRatioCalculator(RatioCalculator):
    """Calculate PR flow ratio = opened / closed."""

    deps = (OpenedCalculator, ClosedCalculator)


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
        sizes = np.repeat(facts[PullRequestFacts.f.size].values[None, :],
                          len(min_times), axis=0).astype(object)
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
        other = ~facts[PullRequestFacts.f.done].values
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


class JIRAMappingCalculator(SumMetricCalculator[int]):
    """Count PRs mapped to JIRA issues."""

    dtype = int

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        result = self._calcs[0].peek.copy()
        result[:, facts[PullRequestFacts.f.jira_id].isnull().values] = None
        return result


class OpenedJIRACalculator(JIRAMappingCalculator):
    """Count PRs mapped to JIRA issues."""

    deps = (OpenedCalculator,)


class DoneJIRACalculator(JIRAMappingCalculator):
    """Count PRs mapped to JIRA issues."""

    deps = (DoneCalculator,)


class AllJIRACalculator(JIRAMappingCalculator):
    """Count PRs mapped to JIRA issues."""

    deps = (AllCounter,)


@register_metric(PullRequestMetricID.PR_OPENED_MAPPED_TO_JIRA)
class OpenedJIRARatioCalculator(RatioCalculator):
    """Calculate opened PRs JIRA mapping ratio = opened and mapped / opened."""

    deps = (OpenedJIRACalculator, OpenedCalculator)


@register_metric(PullRequestMetricID.PR_DONE_MAPPED_TO_JIRA)
class DoneJIRARatioCalculator(RatioCalculator):
    """Calculate done PRs JIRA mapping ratio = done and mapped / done."""

    deps = (DoneJIRACalculator, DoneCalculator)


@register_metric(PullRequestMetricID.PR_ALL_MAPPED_TO_JIRA)
class AllJIRARatioCalculator(RatioCalculator):
    """Calculate all observed PRs JIRA mapping ratio = mapped / all."""

    deps = (AllJIRACalculator, AllCounter)


def need_jira_mapping(metrics: Iterable[str]) -> bool:
    """Check whether some of the metrics require loading the JIRA issue mapping."""
    return bool(set(metrics).intersection({
        PullRequestMetricID.PR_OPENED_MAPPED_TO_JIRA,
        PullRequestMetricID.PR_DONE_MAPPED_TO_JIRA,
        PullRequestMetricID.PR_ALL_MAPPED_TO_JIRA,
    }))
