from datetime import datetime, timedelta
from enum import IntEnum
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Type

import numpy as np
import numpy.typing as npt
import pandas as pd

from athenian.api import metadata
from athenian.api.internal.features.metric import (
    Metric,
    MetricFloat,
    MetricInt,
    MetricTimeDelta,
    T,
    make_metric,
)
from athenian.api.internal.features.metric_calculator import (
    AverageMetricCalculator,
    BinnedHistogramCalculator,
    BinnedMetricCalculator,
    Counter,
    HistogramCalculator,
    HistogramCalculatorEnsemble,
    MedianMetricCalculator,
    MetricCalculator,
    MetricCalculatorEnsemble,
    RatioCalculator,
    SumMetricCalculator,
    ThresholdComparisonRatioCalculator,
    WithoutQuantilesMixin,
    calculate_logical_duplication_mask,
    group_by_lines,
    make_register_metric,
)
from athenian.api.internal.logical_accelerated import drop_logical_in_array
from athenian.api.internal.miners.github.dag_accelerated import searchsorted_inrange
from athenian.api.internal.miners.participation import PRParticipants, PRParticipationKind
from athenian.api.internal.miners.types import DeploymentConclusion, PullRequestFacts
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings
from athenian.api.models.web import PullRequestMetricID
from athenian.api.object_arrays import nested_lengths

metric_calculators: Dict[str, Type[MetricCalculator]] = {}
histogram_calculators: Dict[str, Type[HistogramCalculator]] = {}
register_metric = make_register_metric(metric_calculators, histogram_calculators)


class PullRequestMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float], quantile_stride: int, **kwargs):
        """Initialize a new instance of PullRequestMetricCalculatorEnsemble class."""
        super().__init__(
            *metrics,
            quantiles=quantiles,
            quantile_stride=quantile_stride,
            class_mapping=metric_calculators,
            **kwargs,
        )


class PullRequestHistogramCalculatorEnsemble(HistogramCalculatorEnsemble):
    """HistogramCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float], **kwargs):
        """Initialize a new instance of PullRequestHistogramCalculatorEnsemble class."""
        super().__init__(
            *metrics, quantiles=quantiles, class_mapping=histogram_calculators, **kwargs,
        )


def group_prs_by_lines(lines: Sequence[int], items: pd.DataFrame) -> List[np.ndarray]:
    """
    Bin PRs by number of changed `lines`.

    We throw away the ends: PRs with fewer lines than `lines[0]` and with more lines than \
    `lines[-1]`.

    :param lines: Either an empty sequence or one with at least 2 elements. The numbers must \
                  monotonically increase.
    """
    return group_by_lines(lines, items["size"].values)


def group_prs_by_participants(
    participants: Sequence[PRParticipants],
    items: pd.DataFrame,
) -> List[np.ndarray]:
    """
    Group PRs by participants.

    The aggregation is OR. We don't support all kinds, see `PullRequestFacts`'s mutable fields.
    An array with selected indexes in `items` if returned for every group in `participants`.
    If `participants` is empty a single array selecting everything is returned.
    """
    # FIXME: participants here can also be List[Dict[PRParticipationKind, Set[str]]]

    # if len(participants) == 1, we've already filtered in SQL so don't have to re-check
    # if there are no participant groups also we select everything and don't re-check
    if len(participants) < 2:
        return [np.arange(len(items))]
    # if items is empty we return the right number of indexes to allow subsequent correct grouping
    if items.empty:
        return [np.arange(len(items))] * len(participants)
    groups = []
    log = logging.getLogger("%s.group_prs_by_participants" % metadata.__package__)
    for participants_group in participants:
        group = np.full(len(items), False)
        for participation_kind, devs in participants_group.items():
            name = participation_kind.name.lower()
            if participation_kind in (
                PRParticipationKind.AUTHOR,
                PRParticipationKind.MERGER,
                PRParticipationKind.RELEASER,
            ):
                group |= items[name].isin(devs).values
            else:
                log.warning("Unsupported participation kind: %s", name)
        groups.append(np.nonzero(group)[0])
    return groups


def calculate_logical_prs_duplication_mask(
    items: pd.DataFrame,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
) -> Optional[np.ndarray]:
    """Assign indexes to PRs with the same logical settings for each logical repository."""
    if not logical_settings.has_logical_prs():
        return None
    repos_column = items[PullRequestFacts.f.repository_full_name].values.astype("S", copy=False)
    return calculate_logical_duplication_mask(repos_column, release_settings, logical_settings)


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
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        override_event_time: Optional[datetime] = None,
        override_event_indexes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        wip_end = np.full(len(facts), None, min_times.dtype)
        no_last_review = facts[PullRequestFacts.f.last_review].isnull().values
        has_last_review = ~no_last_review
        frr = facts[PullRequestFacts.f.first_review_request].values
        wip_end[has_last_review] = frr[has_last_review]

        # review was probably requested but never happened
        no_last_commit = facts[PullRequestFacts.f.last_commit].isnull().values
        has_last_commit = ~no_last_commit & no_last_review
        wip_end[has_last_commit] = facts[PullRequestFacts.f.last_commit].values[has_last_commit]

        # 0 commits in the PR, no reviews and review requests
        # => review time = 0
        # => merge time = 0 (you cannot merge an empty PR)
        # => release time = 0
        # This PR is either closed or not fully fetched.
        remaining = np.flatnonzero(np.isnat(wip_end))
        closed = facts[PullRequestFacts.f.closed].values[remaining]
        wip_end[remaining] = closed
        remaining = remaining[closed != closed]
        wip_end[remaining] = frr[remaining]

        if override_event_time is not None:
            wip_end[override_event_indexes] = override_event_time

        wip_end_indexes = np.nonzero(~np.isnat(wip_end))[0]
        dtype = facts[PullRequestFacts.f.created].dtype
        wip_end = wip_end[wip_end_indexes].astype(dtype)
        wip_end_in_range = (min_times[:, None] <= wip_end) & (wip_end < max_times[:, None])
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        work_began = facts[PullRequestFacts.f.work_began].values
        for result_dim, wip_end_in_range_dim in zip(result, wip_end_in_range):
            wip_end_indexes_dim = wip_end_indexes[wip_end_in_range_dim]
            result_dim[wip_end_indexes_dim] = (
                wip_end[wip_end_in_range_dim] - work_began[wip_end_indexes_dim]
            )
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


@register_metric(PullRequestMetricID.PR_WIP_TIME_BELOW_THRESHOLD_RATIO)
class WorkInProgressTimeBelowThresholdRatio(ThresholdComparisonRatioCalculator):
    """Calculate the ratio of PRs with a PR_WIP_TIME below a given threshold."""

    deps = (WorkInProgressTimeCalculator,)
    _compare = np.less_equal
    default_threshold = timedelta(days=1)


@register_metric(PullRequestMetricID.PR_REVIEW_TIME)
class ReviewTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time of the review process metric."""

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        allow_unclosed=False,
        override_event_time: Optional[datetime] = None,
        override_event_indexes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        has_first_review_request = facts[PullRequestFacts.f.first_review_request].notnull().values
        review_end = np.full(len(facts), None, min_times.dtype)
        # we cannot be sure that the approvals finished unless the PR is closed.
        if allow_unclosed:
            closed_mask = has_first_review_request
        else:
            closed_mask = (
                facts[PullRequestFacts.f.closed].notnull().values & has_first_review_request
            )
        not_approved_mask = facts[PullRequestFacts.f.approved].isnull().values
        approved_mask = ~not_approved_mask & closed_mask
        last_review_mask = (
            not_approved_mask
            & facts[PullRequestFacts.f.last_review].notnull().values
            & closed_mask
        )
        review_end[approved_mask] = facts[PullRequestFacts.f.approved].values[approved_mask]
        review_end[last_review_mask] = facts[PullRequestFacts.f.last_review].values[
            last_review_mask
        ]

        if override_event_time is not None:
            review_end[override_event_indexes] = override_event_time

        review_not_none = ~np.isnat(review_end)
        review_in_range = np.full((len(min_times), len(facts)), False)
        dtype = facts[PullRequestFacts.f.created].dtype
        review_end = review_end[review_not_none].astype(dtype)
        review_in_range_mask = (min_times[:, None] <= review_end) & (
            review_end < max_times[:, None]
        )
        review_in_range[:, review_not_none] = review_in_range_mask
        review_end = np.broadcast_to(review_end[None, :], (len(min_times), len(review_end)))
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        frr = facts[PullRequestFacts.f.first_review_request].values
        for result_dim, review_end_dim, review_in_range_mask_dim, review_in_range_dim in zip(
            result, review_end, review_in_range_mask, review_in_range,
        ):
            review_end_dim = review_end_dim[review_in_range_mask_dim]
            result_dim[review_in_range_dim] = review_end_dim - frr[review_in_range_dim]
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


@register_metric(PullRequestMetricID.PR_REVIEW_TIME_BELOW_THRESHOLD_RATIO)
class ReviewTimeBelowThresholdRatio(ThresholdComparisonRatioCalculator):
    """Calculate the ratio of PRs with a PR_REVIEW_TIME below a given threshold."""

    deps = (ReviewTimeCalculator,)
    _compare = np.less_equal
    default_threshold = timedelta(days=2)


@register_metric(PullRequestMetricID.PR_MERGING_TIME)
class MergingTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time to close PR after finishing the review metric."""

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        override_event_time: Optional[datetime] = None,
        override_event_indexes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        merge_end = result.copy().astype(min_times.dtype)
        closed_indexes = np.flatnonzero(facts[PullRequestFacts.f.merged].notnull().values)
        closed = facts[PullRequestFacts.f.merged].values[closed_indexes]
        closed_in_range = (min_times[:, None] <= closed) & (closed < max_times[:, None])
        closed_indexes = np.broadcast_to(
            closed_indexes[None, :], (len(min_times), len(closed_indexes)),
        )
        closed_mask = np.full((len(min_times), len(facts)), False)
        for merge_end_dim, closed_mask_dim, closed_in_range_dim, closed_indexes_dim in zip(
            merge_end, closed_mask, closed_in_range, closed_indexes,
        ):
            closed_indexes_dim = closed_indexes_dim[closed_in_range_dim]
            merge_end_dim[closed_indexes_dim] = closed[closed_in_range_dim]
            closed_mask_dim[closed_indexes_dim] = True

        if override_event_time is not None:
            merge_end[:, override_event_indexes] = override_event_time
            closed_mask[:, override_event_indexes] = True

        dtype = facts[PullRequestFacts.f.created].dtype
        not_approved_mask = np.broadcast_to(
            facts[PullRequestFacts.f.approved].isnull().values[None, :], result.shape,
        )
        approved_mask = ~not_approved_mask & closed_mask
        merge_end_approved = merge_end[approved_mask].astype(dtype)
        approved = np.broadcast_to(
            facts[PullRequestFacts.f.approved].values[None, :], result.shape,
        )
        result[approved_mask] = merge_end_approved - approved[approved_mask]
        not_last_review_mask = np.broadcast_to(
            facts[PullRequestFacts.f.last_review].isnull().values[None, :], result.shape,
        )
        last_review_mask = ~not_last_review_mask & not_approved_mask & closed_mask
        merge_end_last_reviewed = merge_end[last_review_mask].astype(dtype)
        last_review = np.broadcast_to(
            facts[PullRequestFacts.f.last_review].values[None, :], result.shape,
        )
        result[last_review_mask] = merge_end_last_reviewed - last_review[last_review_mask]
        has_last_commit = np.broadcast_to(
            facts[PullRequestFacts.f.last_commit].notnull().values[None, :], result.shape,
        )
        last_commit_mask = not_approved_mask & not_last_review_mask & has_last_commit & closed_mask
        merge_end_last_commit = merge_end[last_commit_mask].astype(dtype)
        last_commit = np.broadcast_to(
            facts[PullRequestFacts.f.last_commit].values[None, :], result.shape,
        )
        result[last_commit_mask] = merge_end_last_commit - last_commit[last_commit_mask]
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


@register_metric(PullRequestMetricID.PR_MERGING_TIME_BELOW_THRESHOLD_RATIO)
class MergingTimeBelowThresholdRatio(ThresholdComparisonRatioCalculator):
    """Calculate the ratio of PRs with a PR_MERGING_TIME below a given threshold."""

    deps = (MergingTimeCalculator,)
    _compare = np.less_equal
    default_threshold = timedelta(hours=4)


@register_metric(PullRequestMetricID.PR_RELEASE_TIME)
class ReleaseTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time to appear in a release after merging metric."""

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        override_event_time: Optional[datetime] = None,
        override_event_indexes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        released = facts[PullRequestFacts.f.released].values
        released_mask = (min_times[:, None] <= released) & (released < max_times[:, None])
        release_end = result.copy().astype(min_times.dtype)
        release_end[released_mask] = np.broadcast_to(released[None, :], result.shape)[
            released_mask
        ]

        if override_event_time is not None:
            release_end[:, override_event_indexes] = override_event_time
            released_mask[:, override_event_indexes] = True

        result_mask = released_mask
        result_mask[:, facts[PullRequestFacts.f.merged].isnull().values] = False
        merged = np.broadcast_to(facts[PullRequestFacts.f.merged].values[None, :], result.shape)[
            result_mask
        ]
        release_end = release_end[result_mask]
        result[result_mask] = release_end - merged
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


@register_metric(PullRequestMetricID.PR_OPEN_TIME)
class OpenTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time the PR stayed open metric."""

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        override_event_time: Optional[datetime] = None,
        override_event_indexes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        closed = facts[PullRequestFacts.f.closed].values
        closed_mask = (min_times[:, None] <= closed) & (closed < max_times[:, None])
        closed = np.broadcast_to(closed[None, :], result.shape)[closed_mask]
        opened = np.broadcast_to(facts[PullRequestFacts.f.created].values[None, :], result.shape)[
            closed_mask
        ]
        result[closed_mask] = closed - opened
        return result


@register_metric(PullRequestMetricID.PR_OPEN_COUNT)
class OpenCounter(WithoutQuantilesMixin, Counter):
    """Count the number of PRs that were used to calculate PR_OPEN_TIME disregarding \
    the quantiles."""

    deps = (OpenTimeCalculator,)


@register_metric(PullRequestMetricID.PR_OPEN_COUNT_Q)
class OpenCounterWithQuantiles(Counter):
    """Count the number of PRs that were used to calculate PR_OPEN_TIME respecting \
    the quantiles."""

    deps = (OpenTimeCalculator,)


@register_metric(PullRequestMetricID.PR_OPEN_TIME_BELOW_THRESHOLD_RATIO)
class OpenTimeBelowThresholdRatio(ThresholdComparisonRatioCalculator):
    """Calculate the ratio of PRs with a PR_OPEN_TIME below a given threshold."""

    deps = (OpenTimeCalculator,)
    _compare = np.less_equal
    default_threshold = timedelta(days=3)


@register_metric(PullRequestMetricID.PR_LEAD_TIME)
@register_metric(PullRequestMetricID.PR_CYCLE_TIME)
class CycleTimeCalculator(AverageMetricCalculator[timedelta]):
    """Time to appear in a release since starting working on the PR."""

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        released_indexes = np.nonzero(facts[PullRequestFacts.f.released].notnull().values)[0]
        released = facts[PullRequestFacts.f.released].values[released_indexes]
        released_in_range = (min_times[:, None] <= released) & (released < max_times[:, None])
        work_began = facts[PullRequestFacts.f.work_began].values
        for result_dim, released_in_range_dim in zip(result, released_in_range):
            released_indexes_dim = released_indexes[released_in_range_dim]
            result_dim[released_indexes_dim] = (
                released[released_in_range_dim] - work_began[released_indexes_dim]
            )
        return result


@register_metric(PullRequestMetricID.PR_LEAD_TIME_BELOW_THRESHOLD_RATIO)
@register_metric(PullRequestMetricID.PR_CYCLE_TIME_BELOW_THRESHOLD_RATIO)
class CycleTimeBelowThresholdRatio(ThresholdComparisonRatioCalculator):
    """Calculate the ratio of PRs with a PR_LEAD_TIME below the threshold."""

    deps = (CycleTimeCalculator,)
    _compare = np.less_equal
    default_threshold = timedelta(days=5)


@register_metric(PullRequestMetricID.PR_LEAD_COUNT)
@register_metric(PullRequestMetricID.PR_CYCLE_COUNT)
class CycleCounter(WithoutQuantilesMixin, Counter):
    """Count the number of PRs that were used to calculate PR_CYCLE_TIME disregarding \
    the quantiles."""

    deps = (CycleTimeCalculator,)


@register_metric(PullRequestMetricID.PR_LEAD_COUNT_Q)
@register_metric(PullRequestMetricID.PR_CYCLE_COUNT_Q)
class CycleCounterWithQuantiles(Counter):
    """Count the number of PRs that were used to calculate PR_CYCLE_TIME respecting \
    the quantiles."""

    deps = (CycleTimeCalculator,)


@register_metric(PullRequestMetricID.PR_LIVE_CYCLE_TIME)
class LiveCycleTimeCalculator(MetricCalculator[timedelta]):
    """Sum of PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGING_TIME, and PR_RELEASE_TIME."""

    deps = (
        WorkInProgressTimeCalculator,
        ReviewTimeCalculator,
        MergingTimeCalculator,
        ReleaseTimeCalculator,
    )
    metric = MetricTimeDelta
    only_complete = False

    def _values(self) -> List[List[MetricTimeDelta]]:
        """Calculate the current metric value."""
        out_shape = self.samples.shape[:2]
        raw_metrics = np.zeros(out_shape, dtype=self.metric.dtype).ravel()
        if self.only_complete:
            raw_metrics["exists"] = True
        for calc in self._calcs:
            calc_values = np.empty(out_shape, dtype=object)
            calc_values[:] = calc.values
            calc_values = np.fromiter(
                (m.array for m in calc_values.ravel()), self.metric.dtype, calc_values.size,
            )
            calc_exists = calc_values["exists"]
            if self.only_complete:
                raw_metrics["exists"][~calc_exists] = False
            else:
                raw_metrics["exists"][calc_exists] = True
            for field in ("value", "confidence_min", "confidence_max"):
                raw_metrics[field][calc_exists] += calc_values[field][calc_exists]
        not_exists = ~raw_metrics["exists"]
        for field in ("value", "confidence_min", "confidence_max"):
            raw_metrics[field][not_exists] = self.nan
        obj_metrics = np.empty(out_shape, dtype=object)
        obj_metrics.ravel()[:] = [self.metric(i) for i in raw_metrics]
        return obj_metrics

    def _value(self, samples: np.ndarray) -> Metric[timedelta]:
        raise AssertionError("this must be never called")

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Update the states of the underlying calcs and return whether at least one of the PR's \
        metrics exists."""
        if self.only_complete:
            sumval = np.zeros((len(min_times), len(facts)), self.dtype)
        else:
            sumval = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        for calc in self._calcs:
            peek = calc.peek
            if self.only_complete:
                sumval += peek
            else:
                sum_none_mask = np.isnat(sumval)
                peek_not_none_mask = ~np.isnat(peek)
                copy_mask = sum_none_mask & peek_not_none_mask
                sumval[copy_mask] = peek[copy_mask]
                add_mask = ~sum_none_mask & peek_not_none_mask
                sumval[add_mask] += peek[add_mask]
        return sumval


@register_metric(PullRequestMetricID.PR_LIVE_CYCLE_COUNT)
class LiveCycleCounter(WithoutQuantilesMixin, Counter):
    """Count unique PRs that were used to calculate PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGING_TIME, \
    or PR_RELEASE_TIME disregarding the quantiles."""

    deps = (LiveCycleTimeCalculator,)


@register_metric(PullRequestMetricID.PR_LIVE_CYCLE_COUNT_Q)
class LiveCycleCounterWithQuantiles(Counter):
    """Count unique PRs that were used to calculate PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGING_TIME, \
    or PR_RELEASE_TIME respecting the quantiles."""

    deps = (LiveCycleTimeCalculator,)

    def _values(self) -> List[List[Metric[T]]]:
        if self._quantiles == (0, 1):
            return super()._values()
        if self._calcs[0].only_complete:
            mask = np.ones(self._calcs[0].grouped_sample_mask.shape, dtype=bool)
            for calc in self._calcs[0]._calcs:
                mask &= calc.grouped_sample_mask.dense()
        else:
            mask = np.zeros(self._calcs[0].grouped_sample_mask.shape, dtype=bool)
            for calc in self._calcs[0]._calcs:
                mask |= calc.grouped_sample_mask.dense()
        return [
            [self.metric.from_fields(True, s, None, None) for s in gs] for gs in mask.sum(axis=-1)
        ]


@register_metric(PullRequestMetricID.PR_ALL_COUNT)
class AllCounter(SumMetricCalculator[int]):
    """Count all the PRs that are active in the given time interval - but ignoring \
    the deployments."""

    metric = MetricInt
    exclude_inactive = False

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        created_in_range_mask = facts[PullRequestFacts.f.created].values < max_times[:, None]
        released = facts[PullRequestFacts.f.released].values
        released_in_range_mask = released >= min_times[:, None]
        closed = facts[PullRequestFacts.f.closed].values
        closed_in_range_mask = (closed >= min_times[:, None]) | (closed != closed)
        if not self.exclude_inactive:
            merged_unreleased_mask = (
                facts[PullRequestFacts.f.merged].values < min_times[:, None]
            ) & (released != released)
        else:
            merged_unreleased_mask = np.array([False])
            # we should intersect each PR's activity days with [min_times, max_times).
            # the following is similar to ReviewedCalculator
            activity_mask = np.full((len(min_times), len(facts)), False)
            activity_days = np.concatenate(facts[PullRequestFacts.f.activity_days].values).astype(
                facts[PullRequestFacts.f.created].dtype,
            )
            activities_in_range = (min_times[:, None] <= activity_days) & (
                activity_days < max_times[:, None]
            )
            activity_offsets = np.zeros(len(facts) + 1, dtype=int)
            np.cumsum(
                facts[PullRequestFacts.f.activity_days].apply(len).values,
                out=activity_offsets[1:],
            )
            for activity_mask_dim, activities_in_range_dim in zip(
                activity_mask, activities_in_range,
            ):
                activity_indexes = np.unique(
                    np.searchsorted(
                        activity_offsets, np.nonzero(activities_in_range_dim)[0], side="right",
                    )
                    - 1,
                )
                activity_mask_dim[activity_indexes] = 1

        in_range_mask = created_in_range_mask & (
            released_in_range_mask | closed_in_range_mask | merged_unreleased_mask
        )
        if self.exclude_inactive:
            in_range_mask &= activity_mask
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        result[in_range_mask] = True
        return result


@register_metric(PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME)
class WaitFirstReviewTimeCalculator(AverageMetricCalculator[timedelta]):
    """Elapsed time between requesting the review for the first time and getting it."""

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        result_mask = (
            facts[PullRequestFacts.f.first_comment_on_first_review].notnull().values
            & facts[PullRequestFacts.f.first_review_request].notnull().values
        )
        fc_on_fr = facts[PullRequestFacts.f.first_comment_on_first_review].values[result_mask]
        fc_on_fr_in_range_mask = (min_times[:, None] <= fc_on_fr) & (fc_on_fr < max_times[:, None])
        result_indexes = np.nonzero(result_mask)[0]
        first_review_request = facts[PullRequestFacts.f.first_review_request].values
        for result_dim, fc_on_fr_in_range_mask_dim in zip(result, fc_on_fr_in_range_mask):
            result_indexes_dim = result_indexes[fc_on_fr_in_range_mask_dim]
            result_dim[result_indexes_dim] = (
                fc_on_fr[fc_on_fr_in_range_mask_dim] - first_review_request[result_indexes_dim]
            )
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


@register_metric(PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME_BELOW_THRESHOLD_RATIO)
class WaitFirstReviewTimeBelowThresholdRatio(ThresholdComparisonRatioCalculator):
    """Calculate the ratio of PRs with a PR_WAIT_FIRST_REVIEW_TIME below a given threshold."""

    deps = (WaitFirstReviewTimeCalculator,)
    _compare = np.less_equal
    default_threshold = timedelta(hours=6)


@register_metric(PullRequestMetricID.PR_OPENED)
class OpenedCalculator(SumMetricCalculator[int]):
    """Number of open PRs."""

    metric = MetricInt

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        created = facts[PullRequestFacts.f.created].values
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        result[(min_times[:, None] <= created) & (created < max_times[:, None])] = 1
        return result


@register_metric(PullRequestMetricID.PR_REVIEWED)
class ReviewedCalculator(SumMetricCalculator[int]):
    """Number of reviewed PRs."""

    metric = MetricInt

    def _analyze(
        self,
        facts: np.ndarray,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        review_timestamps = np.concatenate(facts[PullRequestFacts.f.reviews].values).astype(
            facts[PullRequestFacts.f.created].dtype,
        )
        reviews_in_range = (min_times[:, None] <= review_timestamps) & (
            review_timestamps < max_times[:, None]
        )
        # we cannot sum `reviews_in_range` because there can be several reviews for the same PR
        review_offsets = np.zeros(len(facts) + 1, dtype=int)
        np.cumsum(facts[PullRequestFacts.f.reviews].apply(len).values, out=review_offsets[1:])
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        for result_dim, reviews_in_range_dim in zip(result, reviews_in_range):
            # np.searchsorted aliases several reviews of the same PR to the right border of a
            # `review_offsets` interval
            # np.unique collapses duplicate indexes
            reviewed_indexes = np.unique(
                np.searchsorted(review_offsets, np.nonzero(reviews_in_range_dim)[0], side="right")
                - 1,
            )
            result_dim[reviewed_indexes] = 1
        return result


@register_metric(PullRequestMetricID.PR_MERGED)
class MergedCalculator(SumMetricCalculator[int]):
    """Number of merged PRs."""

    metric = MetricInt

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        merged = facts[PullRequestFacts.f.merged].values
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        result[(min_times[:, None] <= merged) & (merged < max_times[:, None])] = 1
        return result


@register_metric(PullRequestMetricID.PR_REJECTED)
class RejectedCalculator(SumMetricCalculator[int]):
    """Number of rejected PRs."""

    metric = MetricInt

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        closed = facts[PullRequestFacts.f.closed].values
        closed_in_range_mask = (min_times[:, None] <= closed) & (closed < max_times[:, None])
        unmerged_mask = facts[PullRequestFacts.f.merged].isnull().values
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        result[closed_in_range_mask & unmerged_mask] = 1
        return result


@register_metric(PullRequestMetricID.PR_CLOSED)
class ClosedCalculator(SumMetricCalculator[int]):
    """Number of closed PRs."""

    metric = MetricInt

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        closed = facts[PullRequestFacts.f.closed].values
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        result[(min_times[:, None] <= closed) & (closed < max_times[:, None])] = 1
        return result


@register_metric(PullRequestMetricID.PR_NOT_REVIEWED)
class NotReviewedCalculator(SumMetricCalculator[int]):
    """Number of non-reviewed closed PRs."""

    deps = (ReviewedCalculator, ClosedCalculator)
    metric = MetricInt

    def _analyze(
        self,
        facts: np.ndarray,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        not_reviewed = self._calcs[0].peek == self.nan
        result = not_reviewed.astype(self.dtype)
        result[(self._calcs[1].peek == self.nan) | ~not_reviewed] = self.nan
        return result


@register_metric(PullRequestMetricID.PR_REVIEWED_CLOSED)
class ReviewedClosedCalculator(SumMetricCalculator[int]):
    """Number of closed PRs that were reviewed."""

    deps = (ReviewedCalculator, ClosedCalculator)
    metric = MetricInt

    def _analyze(
        self,
        facts: np.ndarray,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        result = self._calcs[0].peek.copy()
        result[self._calcs[1].peek == self.nan] = self.nan
        return result


@register_metric(PullRequestMetricID.PR_DONE)
class DoneCalculator(SumMetricCalculator[int]):
    """Number of rejected or released PRs."""

    metric = MetricInt

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        released = facts[PullRequestFacts.f.released].values
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        result[(min_times[:, None] <= released) & (released < max_times[:, None])] = 1
        rejected_mask = facts[PullRequestFacts.f.closed].notnull().values & (
            facts[PullRequestFacts.f.merged].isnull().values
            | facts[PullRequestFacts.f.force_push_dropped].values
            | facts[PullRequestFacts.f.release_ignored].values
        )
        closed = facts[PullRequestFacts.f.closed].values
        result[(min_times[:, None] <= closed) & (closed < max_times[:, None]) & rejected_mask] = 1
        return result


class _ReviewedPlusNotReviewedCalculator(MetricCalculator[int]):
    """Calculate the sum reviewed + non-reviewed PRs.

    This metric is not exposed but only used to compute PR_REVIEWED_RATIO.
    """

    deps = (ReviewedCalculator, NotReviewedCalculator)
    metric = MetricInt

    def _values(self) -> list[list[Metric[int]]]:
        metrics = [
            [self.metric.from_fields(False, None, None, None)] * len(samples)
            for samples in self.samples
        ]
        a, b = self._calcs
        for i, (a_group, b_group) in enumerate(zip(a.values, b.values)):
            for j, (a_metric, b_metric) in enumerate(zip(a_group, b_group)):
                tot = 0
                if a_metric.exists or b_metric.exists:
                    tot += (a_metric.value or 0) + (b_metric.value or 0)
                    metrics[i][j] = self.metric.from_fields(True, tot, None, None)

        return metrics

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return np.full((len(min_times), len(facts)), self.nan, self.dtype)

    def _value(self, samples: np.ndarray) -> Metric[timedelta]:
        raise AssertionError("this must be never called")


@register_metric(PullRequestMetricID.PR_REVIEWED_RATIO)
class ReviewedRatioCalculator(RatioCalculator):
    """Calculate the PR reviewed ratio = pr-reviewed / (pr-reviewed + pr-not-reviewed)."""

    deps = (ReviewedCalculator, _ReviewedPlusNotReviewedCalculator)


@register_metric(PullRequestMetricID.PR_FLOW_RATIO)
class FlowRatioCalculator(RatioCalculator):
    """Calculate PR flow ratio = opened / closed."""

    deps = (OpenedCalculator, ClosedCalculator)
    value_offset = 1


class SizeCalculatorMixin(MetricCalculator[int]):
    """Calculate any aggregating statistic over PR sizes."""

    may_have_negative_values = False
    metric = MetricInt

    def _shift_log(self, samples: np.ndarray) -> np.ndarray:
        samples[samples == 0] = 1
        return samples

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        sizes = np.repeat(facts[PullRequestFacts.f.size].values[None, :], len(min_times), axis=0)
        created = facts[PullRequestFacts.f.created].values
        sizes[~((min_times[:, None] <= created) & (created < max_times[:, None]))] = self.nan
        return sizes


@register_metric(PullRequestMetricID.PR_SIZE)
class SizeCalculator(SizeCalculatorMixin, AverageMetricCalculator[int]):
    """Average PR size."""


@register_metric(PullRequestMetricID.PR_MEDIAN_SIZE)
class MedianSizeCalculator(SizeCalculatorMixin, MedianMetricCalculator[int]):
    """Median PR size."""


@register_metric(PullRequestMetricID.PR_SIZE_BELOW_THRESHOLD_RATIO)
class SizeBelowThresholdRatio(ThresholdComparisonRatioCalculator):
    """Calculate the ratio of PRs with a size below a given threshold."""

    deps = (SizeCalculator,)
    _compare = np.less_equal
    default_threshold = 100


class PendingStage(IntEnum):
    """Indexes of the pending stages that are used below."""

    WIP = 0
    REVIEW = 1
    MERGE = 2
    RELEASE = 3


class StagePendingDependencyCalculator(SumMetricCalculator[int]):
    """Common dependency for stage-pending counters."""

    metric = MetricInt
    deps = (AllCounter,)
    is_pure_dependency = True

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        merged_mask, approved_mask, frr_mask = (
            facts[c].notnull().values for c in ("merged", "approved", "first_review_request")
        )

        stage_masks = np.zeros((len(min_times), len(facts), len(PendingStage)), self.dtype)
        other = ~facts[PullRequestFacts.f.done].values
        stage_masks[:, merged_mask & other, PendingStage.RELEASE] = True
        other &= ~merged_mask
        stage_masks[:, approved_mask & other, PendingStage.MERGE] = True
        other &= ~approved_mask
        stage_masks[:, frr_mask & other, PendingStage.REVIEW] = True
        other &= ~frr_mask
        stage_masks[:, other, PendingStage.WIP] = True
        stage_masks[self._calcs[0].peek == self._calcs[0].nan] = False
        return stage_masks


class BaseStagePendingCounter(SumMetricCalculator[int]):
    """Base stage-pending counter."""

    metric = MetricInt
    stage = None
    deps = (StagePendingDependencyCalculator,)

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
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

    metric = MetricInt

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        result = self._calcs[0].peek.copy()
        result[
            :,
            ~nested_lengths(facts[PullRequestFacts.INDIRECT_FIELDS.JIRA_IDS].values).astype(bool),
        ] = self.nan
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
    return bool(
        set(metrics).intersection(
            {
                PullRequestMetricID.PR_OPENED_MAPPED_TO_JIRA,
                PullRequestMetricID.PR_DONE_MAPPED_TO_JIRA,
                PullRequestMetricID.PR_ALL_MAPPED_TO_JIRA,
            },
        ),
    )


@register_metric(PullRequestMetricID.PR_PARTICIPANTS_PER)
class AverageParticipantsCalculator(AverageMetricCalculator[np.float32]):
    """Average number of PR participants metric."""

    deps = (AllCounter,)
    may_have_negative_values = False
    metric = MetricFloat

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        result = np.repeat(
            facts[PullRequestFacts.f.participants].values[None, :].astype(self.dtype),
            len(min_times),
            axis=0,
        )
        result[self._calcs[0].peek == self._calcs[0].nan] = None
        return result


@register_metric(PullRequestMetricID.PR_REVIEW_COMMENTS_PER)
class AverageReviewCommentsCalculator(AverageMetricCalculator[np.float32]):
    """Average number of PR review comments in reviewed PRs metric."""

    deps = (AllCounter,)
    may_have_negative_values = False
    metric = MetricFloat

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        comments = facts[PullRequestFacts.f.review_comments].values
        empty_mask = comments == 0
        comments = comments.astype(self.dtype)
        comments[empty_mask] = None
        result = np.repeat(comments[None, :], len(min_times), axis=0)
        result[self._calcs[0].peek == self._calcs[0].nan] = None
        return result


@register_metric(PullRequestMetricID.PR_REVIEW_COMMENTS_PER_ABOVE_THRESHOLD_RATIO)
class ReviewCommentsAboveThresholdRatio(ThresholdComparisonRatioCalculator):
    """Calculate the ratio of PRs with a number of review comments above a given threshold."""

    deps = (AverageReviewCommentsCalculator,)
    _compare = np.greater_equal
    default_threshold = 3


@register_metric(PullRequestMetricID.PR_REVIEWS_PER)
class AverageReviewsCalculator(AverageMetricCalculator[np.float32]):
    """Average number of reviews in reviewed PRs metric."""

    deps = (AllCounter,)
    may_have_negative_values = False
    metric = MetricFloat

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        lengths = facts[PullRequestFacts.f.reviews].map(len).values
        empty_mask = lengths == 0
        lengths = lengths.astype(self.dtype)
        lengths[empty_mask] = None
        result = np.repeat(lengths[None, :], len(min_times), axis=0)
        result[self._calcs[0].peek == self._calcs[0].nan] = None
        return result


@register_metric(PullRequestMetricID.PR_COMMENTS_PER)
class AverageCommentsCalculator(AverageMetricCalculator[np.float32]):
    """Average number of PR comments (regular and review) in reviewed PRs metric."""

    deps = (AllCounter,)
    may_have_negative_values = False
    metric = MetricFloat

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        regular_comments = facts[PullRequestFacts.f.regular_comments].values
        review_comments = facts[PullRequestFacts.f.review_comments].values
        comments = regular_comments + review_comments
        empty_mask = review_comments == 0  # only reviewed PRs!
        comments = comments.astype(self.dtype)
        comments[empty_mask] = None
        result = np.repeat(comments[None, :], len(min_times), axis=0)
        result[self._calcs[0].peek == self._calcs[0].nan] = None
        return result


EnvironmentsMarkerDType = np.dtype(
    [
        ("environments", np.ndarray),
        ("counts", np.ndarray),
        ("successful", np.ndarray),
        ("conclusions", np.ndarray),
        ("finished", np.ndarray),
    ],
)
EnvironmentsMarkerMetric = make_metric(
    "EnvironmentsMarkerMetric", __name__, EnvironmentsMarkerDType, None,
)


class EnvironmentsMarker(MetricCalculator[np.ndarray]):
    """Set bits for each PR depending in which deployments it appears."""

    metric = EnvironmentsMarkerMetric
    is_pure_dependency = True
    environments = []

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        assert len(self.environments) <= 64
        result = np.recarray(shape=(), dtype=self.dtype)
        finished_by_env = np.empty(len(self.environments), dtype=object)
        counts_by_env = np.empty_like(finished_by_env)
        conclusions_by_env = np.empty_like(finished_by_env)
        successful_by_env = np.empty_like(finished_by_env)
        counts_by_env.fill(np.array([], dtype=int))
        conclusions_by_env.fill(np.array([], dtype=int))
        successful_by_env.fill(np.array([], dtype=int))
        finished_by_env.fill(np.array([], dtype="datetime64[s]"))
        result.fill(
            (
                np.zeros(len(facts), dtype=np.uint64),
                counts_by_env,
                successful_by_env,
                conclusions_by_env,
                finished_by_env,
            ),
        )
        if facts.empty:
            return result
        envs = np.array(self.environments, dtype="U")
        fact_envs = facts[PullRequestFacts.f.environments].values
        all_envs = np.concatenate(fact_envs).astype("U", copy=False)
        if len(all_envs) == 0:
            return result
        all_finished = np.concatenate(
            facts[PullRequestFacts.f.deployed].values,
            dtype=facts[PullRequestFacts.f.created].dtype,
            casting="unsafe",
        ).astype("datetime64[s]", copy=False)
        all_conclusions = np.concatenate(
            facts[PullRequestFacts.f.deployment_conclusions].values,
            dtype=np.int8,
            casting="unsafe",
        )
        assert len(all_envs) == len(all_finished) == len(all_conclusions)
        unique_fact_envs, imap = np.unique(all_envs, return_inverse=True)
        del all_envs
        unique_fact_envs = unique_fact_envs.astype(
            f"U{np.char.str_len(unique_fact_envs).max()}", copy=False,
        )
        my_env_indexes = searchsorted_inrange(unique_fact_envs, envs)
        not_found_mask = unique_fact_envs[my_env_indexes] != envs
        my_env_indexes[not_found_mask] = np.arange(-1, -1 - not_found_mask.sum(), -1)
        imap = imap.astype(np.uint64)
        checked_mask = np.in1d(
            imap,
            np.setdiff1d(np.arange(len(unique_fact_envs)), my_env_indexes, assume_unique=True),
        )
        imap[checked_mask] = 0

        lengths = nested_lengths(fact_envs)
        offsets = np.zeros(len(lengths) + 1, dtype=int)
        np.cumsum(lengths, out=offsets[1:])
        offsets = offsets[: np.argmax(offsets)]
        no_deps = lengths == 0
        successful_conclusions = all_conclusions == DeploymentConclusion.SUCCESS

        for pos, ix in enumerate(my_env_indexes):
            if ix < 0:
                continue
            ix_mask = imap == ix
            ix_mask[checked_mask] = False
            checked_mask[ix_mask] = True
            imap[ix_mask] = 1 << pos
            finished_by_env[pos] = all_finished[ix_mask]
            # one PR should not fail to deploy more than (1 << 16) times, seems legit
            counts = np.add.reduceat(ix_mask, offsets).astype(np.uint16)
            counts[no_deps[: len(counts)]] = 0
            counts = counts[counts > 0]
            counts_by_env[pos] = counts
            internal_offsets = np.zeros(len(counts), dtype=int)
            np.cumsum(counts[:-1], out=internal_offsets[1:])
            conclusions_by_env[pos] = all_conclusions[ix_mask]
            successful_by_env[pos] = np.bitwise_or.reduceat(
                successful_conclusions[ix_mask], internal_offsets,
            )

        env_marks = np.bitwise_or.reduceat(imap, offsets)
        env_marks[no_deps[: len(env_marks)]] = 0
        env_marks = np.pad(env_marks, (0, len(lengths) - len(env_marks)))
        result.environments.item()[:] = env_marks
        return result

    def _value(self, samples: np.ndarray) -> EnvironmentsMarkerMetric:
        raise AssertionError("this must be never called")


class DeploymentMetricBase(MetricCalculator[T]):
    """Base class of all deployment metric calculators."""

    environment = None
    environments = []
    deps = (EnvironmentsMarker,)

    def split(self) -> List["MetricCalculator"]:
        """Clone yourself as many times as there are environments."""
        assert self.environment is None, "already split"
        if len(self.environments) == 0:
            raise ValueError("Must specify at least one environment.")
        if len(self.environments) == 1:
            if self.environments == [None]:
                return []
            self.environment = 0
            return [self]
        clones = [
            self.clone(
                *(
                    (calcs[i] if isinstance(calcs, (list, tuple)) else calcs)
                    for calcs in self._calcs
                ),
                quantiles=self._quantiles,
                environment=i,
            )
            for i in range(len(self.environments))
        ]
        return clones

    def calc_deployed(
        self,
        facts: Optional[pd.DataFrame] = None,
        columns: Optional[tuple[npt.NDArray[np.timedelta64], npt.NDArray[int]]] = None,
    ) -> npt.NDArray[np.timedelta64]:
        """
        Return the first successful deployment time for each PR in the analyzed `facts`.

        NaT corresponds to 0 successful deployments.
        """
        peek = self._calcs[0].peek
        finished = np.full(len(peek.environments.item()), None, "datetime64[s]")
        nnz_finished = peek.finished.item()[self.environment]
        if nnz_finished is not None:
            mask = (peek.environments.item() & (1 << self.environment)).astype(bool)
            indexes = np.flatnonzero(mask)[peek.successful.item()[self.environment]]
            successful_sum = len(indexes)
            successful = peek.conclusions.item()[self.environment] == DeploymentConclusion.SUCCESS
            nnz_finished = nnz_finished[successful]
            assert successful_sum == len(nnz_finished), (
                f"some PRs deployed more than once in {self.environments[self.environment]}: "
                f"{successful_sum} vs. {len(nnz_finished)}"
            )
            finished[indexes] = nnz_finished
        columns = columns or (
            facts[PullRequestFacts.f.merged].values,
            facts[PullRequestFacts.f.node_id].values,
        )
        merged, pr_node_ids = columns
        mask = finished < merged
        if (contradictions := mask.sum()) > 0:
            log = logging.getLogger(f"{metadata.__package__}.{type(self).__name__}")
            log.error(
                "%d PRs are deployed before they were merged%s",
                contradictions,
                f": {pr_node_ids[mask].tolist()}" if contradictions < 10 else "",
            )
            finished[mask] = self.nan
        return finished


@register_metric(PullRequestMetricID.PR_DEPLOYMENT_TIME)
class DeploymentTimeCalculator(DeploymentMetricBase, AverageMetricCalculator[timedelta]):
    """
    Time between Released and Deployed.

    If Released is later than Deployed, equals to Deployed - Merged.
    If Released does not exist, equals to Deployed - Merged.
    """

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        override_event_time: Optional[datetime] = None,
        override_event_indexes: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        finished = self.calc_deployed(facts)
        if override_event_time is not None:
            finished[override_event_indexes] = override_event_time
        started = facts[PullRequestFacts.f.merged].values.copy()
        released = facts[PullRequestFacts.f.released].values
        release_exists = released <= finished
        started[release_exists] = released[release_exists]
        finished_in_range = (min_times[:, None] <= finished) & (finished < max_times[:, None])
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        delta = finished - started
        result[finished_in_range] = np.broadcast_to(delta[None, :], result.shape)[
            finished_in_range
        ]
        return result


@register_metric(PullRequestMetricID.PR_DEPLOYMENT_COUNT)
@register_metric(PullRequestMetricID.PR_LEAD_DEPLOYMENT_COUNT)
@register_metric(PullRequestMetricID.PR_CYCLE_DEPLOYMENT_COUNT)
class DeploymentCounter(DeploymentMetricBase, WithoutQuantilesMixin, Counter):
    """Count the number of PRs that were used to calculate PR_DEPLOYMENT_TIME \
    disregarding the quantiles."""

    deps = (DeploymentTimeCalculator,)


class DeploymentPendingMarker(DeploymentMetricBase):
    """Auxiliary routine to indicate undeployed PRs belonging to \
    self.repositories[self.environment]."""

    is_pure_dependency = True
    metric = MetricInt
    repositories: List[Sequence[str]] = []
    drop_logical: bool = False

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        finished = self.calc_deployed(facts)
        repo_names = facts[PullRequestFacts.f.repository_full_name].values
        if self.drop_logical:
            repo_names = drop_logical_in_array(repo_names)
        repo_mask = np.in1d(repo_names, self.repositories[self.environment])
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        finished_in_range = finished < max_times[:, None]
        result[:, repo_mask] = 1
        result[finished_in_range] = 0
        return result

    def _value(self, samples: np.ndarray) -> Metric[int]:
        raise AssertionError("this must be never called")


@register_metric(PullRequestMetricID.PR_DEPLOYMENT_COUNT_Q)
class DeploymentCounterWithQuantiles(DeploymentMetricBase, Counter):
    """Count the number of PRs that were used to calculate PR_DEPLOYMENT_TIME respecting \
    the quantiles."""

    deps = (DeploymentTimeCalculator,)


@register_metric(PullRequestMetricID.PR_LEAD_DEPLOYMENT_TIME)
class LeadDeploymentTimeCalculator(DeploymentMetricBase, AverageMetricCalculator[timedelta]):
    """Sum of the regular Lead Time + Deployment Time."""

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        finished = self.calc_deployed(facts)
        work_began = facts[PullRequestFacts.f.work_began].values
        delta = finished - work_began
        finished_in_range = (min_times[:, None] <= finished) & (finished < max_times[:, None])
        result[finished_in_range] = np.broadcast_to(delta, result.shape)[finished_in_range]
        return result


@register_metric(PullRequestMetricID.PR_LEAD_DEPLOYMENT_COUNT_Q)
class LeadDeploymentCounterWithQuantiles(DeploymentMetricBase, Counter):
    """Count the number of PRs that were used to calculate PR_LEAD_DEPLOYMENT_TIME respecting \
    the quantiles."""

    deps = (LeadDeploymentTimeCalculator,)


@register_metric(PullRequestMetricID.PR_CYCLE_DEPLOYMENT_TIME)
class CycleDeploymentTimeCalculator(DeploymentMetricBase, LiveCycleTimeCalculator):
    """Sum of PR_WIP_TIME, PR_REVIEW_TIME, PR_MERGING_TIME, PR_RELEASE_TIME, and \
    PR_DEPLOYMENT_TIME."""

    deps = (LiveCycleTimeCalculator, DeploymentTimeCalculator)
    only_complete = True


@register_metric(PullRequestMetricID.PR_CYCLE_DEPLOYMENT_TIME_BELOW_THRESHOLD_RATIO)
class CycleDeploymentTimeBelowThresholdRatio(ThresholdComparisonRatioCalculator):
    """Calculate the ratio of PRs with a PR_CYCLE_DEPLOYMENT_TIME below a given threshold."""

    deps = (CycleDeploymentTimeCalculator,)
    _compare = np.less_equal
    default_threshold = timedelta(days=5)


@register_metric(PullRequestMetricID.PR_CYCLE_DEPLOYMENT_COUNT_Q)
class CycleDeploymentCounterWithQuantiles(DeploymentMetricBase, LiveCycleCounterWithQuantiles):
    """Count the number of PRs that were used to calculate PR_CYCLE_DEPLOYMENT_TIME respecting \
    the quantiles."""

    deps = (CycleDeploymentTimeCalculator,)
