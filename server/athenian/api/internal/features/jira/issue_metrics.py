from datetime import timedelta
from typing import Dict, List, Sequence, Type

import numpy as np
import pandas as pd

from athenian.api.internal.features.metric import MetricInt, MetricTimeDelta
from athenian.api.internal.features.metric_calculator import (
    AverageMetricCalculator,
    BinnedHistogramCalculator,
    BinnedMetricCalculator,
    Counter,
    HistogramCalculator,
    HistogramCalculatorEnsemble,
    MetricCalculator,
    MetricCalculatorEnsemble,
    RatioCalculator,
    SumMetricCalculator,
    ThresholdComparisonRatioCalculator,
    WithoutQuantilesMixin,
    make_register_metric,
)
from athenian.api.internal.miners.filters import LabelFilter
from athenian.api.internal.miners.jira.issue import ISSUE_PRS_BEGAN, ISSUE_PRS_RELEASED
from athenian.api.internal.miners.types import JIRAParticipants, JIRAParticipationKind
from athenian.api.models.metadata.jira import AthenianIssue, Issue, Status
from athenian.api.models.web import JIRAMetricID
from athenian.api.unordered_unique import in1d_str, unordered_unique

metric_calculators: Dict[str, Type[MetricCalculator]] = {}
histogram_calculators: Dict[str, Type[HistogramCalculator]] = {}
register_metric = make_register_metric(metric_calculators, histogram_calculators)


class JIRAMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for JIRA issues."""

    def __init__(self, *metrics: str, quantiles: Sequence[float], quantile_stride: int):
        """Initialize a new instance of JIRAMetricCalculatorEnsemble class."""
        super().__init__(
            *metrics,
            quantiles=quantiles,
            quantile_stride=quantile_stride,
            class_mapping=metric_calculators,
        )


class JIRAHistogramCalculatorEnsemble(HistogramCalculatorEnsemble):
    """HistogramCalculatorEnsemble adapted for JIRA issues."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of JIRAHistogramCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=histogram_calculators)


class JIRABinnedMetricCalculator(BinnedMetricCalculator):
    """BinnedMetricCalculator adapted for JIRA issues."""

    ensemble_class = JIRAMetricCalculatorEnsemble


class JIRABinnedHistogramCalculator(BinnedHistogramCalculator):
    """BinnedHistogramCalculator adapted for JIRA issues."""

    ensemble_class = JIRAHistogramCalculatorEnsemble


def split_issues_by_participants(
    with_: List[JIRAParticipants],
    issues: pd.DataFrame,
) -> List[np.ndarray]:
    """Group issues by participants."""
    result = []
    if len(with_) < 2:
        # supposing that we fetched only those issues that relate to the only group
        return [np.arange(len(issues))]
    for group in with_:
        mask = np.full(len(issues), False)
        if assignees := group.get(JIRAParticipationKind.ASSIGNEE):
            # None will become "None" and will match; nobody is going to name a user "None"
            # except for to troll Athenian.
            assignees = np.char.lower(np.array(assignees, dtype="U"))
            mask |= in1d_str(issues["assignee"].values.astype("U"), assignees)
        if reporters := group.get(JIRAParticipationKind.REPORTER):
            reporters = np.char.lower(np.array(reporters, dtype="U"))
            mask |= in1d_str(issues["reporter"].values.astype("U"), reporters)
        if commenters := group.get(JIRAParticipationKind.COMMENTER):
            commenters = np.char.lower(np.array(commenters, dtype="U"))
            issue_commenters = issues["commenters"]
            merged_issue_commenters = np.concatenate(issue_commenters).astype("U")
            offsets = np.zeros(len(issue_commenters) + 1, dtype=int)
            np.cumsum(issue_commenters.apply(len).values, out=offsets[1:])
            indexes = unordered_unique(
                np.searchsorted(
                    offsets,
                    np.nonzero(np.in1d(merged_issue_commenters, commenters))[0],
                    side="right",
                )
                - 1,
            )
            mask[indexes] = True
        result.append(np.flatnonzero(mask))
    return result


class IssuesLabelSplitter:
    """Grouper of JIRA issues by labels."""

    def __init__(self, enabled: bool, label_filter: LabelFilter):
        """Initialize a new instance of IssuesLabelSplitter."""
        self._labels = np.array([None], dtype=object)
        self.enabled = enabled
        self._filter = label_filter

    @property
    def labels(self) -> np.ndarray:
        """Return the labels filter value."""
        return self._labels

    def __call__(self, issues: pd.DataFrame) -> List[np.ndarray]:
        """Perform the issues grouping by labels."""
        if not self.enabled or issues.empty:
            return [np.arange(len(issues))]
        labels_column = issues[Issue.labels.name].values
        rows_all_labels = np.repeat(
            np.arange(len(labels_column), dtype=int), [len(labels) for labels in labels_column],
        )
        all_labels = np.concatenate(labels_column).astype("U")
        del labels_column
        all_labels_order = np.argsort(all_labels)
        ordered_rows_all_labels = rows_all_labels[all_labels_order]
        unique_labels, unique_counts = np.unique(all_labels[all_labels_order], return_counts=True)
        del all_labels
        groups = np.array(
            np.split(ordered_rows_all_labels, np.cumsum(unique_counts)), dtype=object,
        )
        unique_labels_order = np.argsort(-unique_counts)
        unique_labels = unique_labels[unique_labels_order]
        groups = groups[unique_labels_order]
        if self._filter.exclude and len(unique_labels):
            exclude = np.array(sorted(self._filter.exclude), dtype="U")
            mask = np.in1d(unique_labels, exclude, assume_unique=True, invert=True)
            unique_labels = unique_labels[mask]
            groups = groups[mask]
        if self._filter.include:
            if len(unique_labels):
                singles, multiples = LabelFilter.split(self._filter.include)
                include = set(singles)
                for labels in multiples:
                    include.update(labels)
                include = np.array(sorted(self._filter.include), dtype="U")
                mask = np.in1d(unique_labels, include, assume_unique=True)
                unique_labels = unique_labels[mask]
                groups = groups[mask]
        else:
            # no include filter => append another group of issues with empty labels
            unique_labels = np.concatenate([unique_labels, [None]])
            empty_labels_group = np.nonzero(~issues[Issue.labels.name].astype(bool).values)[0]
            groups = list(groups) + [empty_labels_group]
        if not isinstance(groups, list):
            groups = groups.tolist()
        self._labels = unique_labels
        return groups


@register_metric(JIRAMetricID.JIRA_RAISED)
class RaisedCounter(SumMetricCalculator[int]):
    """Number of created issues metric."""

    metric = MetricInt

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **_,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        created = facts[Issue.created.name].values
        result[(min_times[:, None] <= created) & (created < max_times[:, None])] = 1
        return result


@register_metric(JIRAMetricID.JIRA_RESOLVED)
class ResolvedCounter(SumMetricCalculator[int]):
    """Number of resolved issues metric."""

    metric = MetricInt

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **_,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        resolved = facts[AthenianIssue.resolved.name].values.astype(min_times.dtype)
        prs_began = facts[ISSUE_PRS_BEGAN].values.astype(min_times.dtype)
        have_prs_mask = prs_began == prs_began
        released = facts[ISSUE_PRS_RELEASED].values.astype(min_times.dtype)[have_prs_mask]
        resolved[have_prs_mask][released != released] = np.datetime64("nat")
        result[(min_times[:, None] <= resolved) & (resolved < max_times[:, None])] = 1
        return result


@register_metric(JIRAMetricID.JIRA_OPEN)
class OpenCounter(SumMetricCalculator[int]):
    """Number of created issues metric."""

    metric = MetricInt

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **_,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        created = facts[Issue.created.name].values
        resolved = facts[AthenianIssue.resolved.name].values.astype(min_times.dtype)
        prs_began = facts[ISSUE_PRS_BEGAN].values.astype(min_times.dtype)
        have_prs_mask = prs_began == prs_began
        released = facts[ISSUE_PRS_RELEASED].values.astype(min_times.dtype)[have_prs_mask]
        resolved[have_prs_mask][released != released] = np.datetime64("nat")
        not_resolved = resolved != resolved
        resolved_later = resolved >= max_times[:, None]
        created_earlier = created < max_times[:, None]
        result[(resolved_later | not_resolved) & created_earlier] = 1
        return result


@register_metric(JIRAMetricID.JIRA_RESOLUTION_RATE)
class ResolutionRateCalculator(RatioCalculator):
    """Calculate JIRA issues flow ratio = raised / resolved."""

    deps = (ResolvedCounter, RaisedCounter)


@register_metric(JIRAMetricID.JIRA_LIFE_TIME)
class LifeTimeCalculator(AverageMetricCalculator[timedelta]):
    """
    Issue Life Time calculator.

    Life Time is the time it takes for a ticket to go from the ticket creation to release.

    * If an issue is linked to PRs, the MTTR ends at the last PR release.
    * If an issue is not linked to any PR, the MTTR ends when the issue transition to the Done \
    status category.
    * If an issue is created after the work began, we consider the latter as the real ticket \
    creation.
    """

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **_,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, self.dtype)
        created = facts[Issue.created.name].values
        prs_began = facts[ISSUE_PRS_BEGAN].values.astype(min_times.dtype)
        resolved = facts[Issue.resolved.name].values.astype(min_times.dtype)
        released = facts[ISSUE_PRS_RELEASED].values.astype(min_times.dtype)
        focus_mask = (min_times[:, None] <= resolved) & (resolved < max_times[:, None])
        life_times = np.maximum(released, resolved) - np.fmin(created, prs_began)
        unmapped_mask = prs_began != prs_began
        life_times[unmapped_mask] = resolved[unmapped_mask] - created[unmapped_mask]
        result[:] = life_times
        result[~focus_mask] = None
        return result


@register_metric(JIRAMetricID.JIRA_LEAD_TIME)
class LeadTimeCalculator(AverageMetricCalculator[timedelta]):
    """
    Issue Lead Time calculator.

    Lead time is the time it takes for the changes to be released since the work on them started.

    * If an issue is linked to PRs, the MTTR ends at the last PR release.
    * If an issue is not linked to any PR, the MTTR ends when the issue transition to the Done
    status category.
    * The timestamp of work_began is min(issue became in progress, PR created).
    """

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **_,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, self.dtype)
        work_began = facts[AthenianIssue.work_began.name].values.astype(min_times.dtype)
        prs_began = facts[ISSUE_PRS_BEGAN].values.astype(min_times.dtype)
        resolved = facts[Issue.resolved.name].values.astype(min_times.dtype)
        released = facts[ISSUE_PRS_RELEASED].values.astype(min_times.dtype)
        focus_mask = (min_times[:, None] <= resolved) & (resolved < max_times[:, None])
        lead_times = np.maximum(released, resolved) - np.fmin(work_began, prs_began)
        unmapped_mask = prs_began != prs_began
        lead_times[unmapped_mask] = resolved[unmapped_mask] - work_began[unmapped_mask]
        result[:] = lead_times
        result[~focus_mask] = None
        return result


@register_metric(JIRAMetricID.JIRA_LEAD_TIME_BELOW_THRESHOLD_RATIO)
class LeadTimeBelowThresholdRatio(ThresholdComparisonRatioCalculator):
    """Calculate the ratio of issues with a JIRA_LEAD_TIME below a given threshold."""

    deps = (LeadTimeCalculator,)
    compare = np.less_equal
    default_threshold = timedelta(days=5)


@register_metric(JIRAMetricID.JIRA_ACKNOWLEDGE_TIME)
class AcknowledgeTimeCalculator(AverageMetricCalculator[timedelta]):
    """
    Issue Acknowledge Time calculator.

    Acknowledge time is the time it takes for the work to actually start after the issue is \
    created. We calculate the metric only for those issues that are currently In Progress or \
    Resolved.

    * If the work began before the creation time, the acknowledge time is 0.
    """

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **_,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, self.dtype)
        work_began = facts[AthenianIssue.work_began.name].values.astype(min_times.dtype)
        prs_began = facts[ISSUE_PRS_BEGAN].values.astype(min_times.dtype)
        acknowledged = np.fmin(work_began, prs_began)
        statuses = facts[Status.category_name.name].values
        acknowledged[
            (statuses != Status.CATEGORY_IN_PROGRESS) & (statuses != Status.CATEGORY_DONE)
        ] = None
        created = facts[Issue.created.name].values.astype(min_times.dtype)
        ack_times = acknowledged - created
        ack_times = np.maximum(ack_times, ack_times.dtype.type(0))
        focus_mask = (min_times[:, None] <= acknowledged) & (acknowledged < max_times[:, None])
        result[:] = ack_times
        result[~focus_mask] = None
        return result


@register_metric(JIRAMetricID.JIRA_ACKNOWLEDGED)
class AcknowledgedCounter(WithoutQuantilesMixin, Counter):
    """Count the number of PRs that were used to calculate JIRA_ACKNOWLEDGED disregarding \
    the quantiles."""

    deps = (AcknowledgeTimeCalculator,)


@register_metric(JIRAMetricID.JIRA_ACKNOWLEDGED_Q)
class AcknowledgedWithQuantiles(Counter):
    """Count the number of PRs that were used to calculate JIRA_ACKNOWLEDGED respecting \
    the quantiles."""

    deps = (AcknowledgeTimeCalculator,)


@register_metric(JIRAMetricID.JIRA_PR_LAG_TIME)
class PRLagTimeCalculator(AverageMetricCalculator[timedelta]):
    """
    Issue PR Lag Time calculator.

    The time between when the issue transitioned to "In Progress" and the first PR was created.
    It equals to 0 if the difference is negative or no PRs are mapped to the issue.
    """

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **_,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, self.dtype)
        work_began = facts[AthenianIssue.work_began.name].values.astype(min_times.dtype)
        prs_began = facts[ISSUE_PRS_BEGAN].values.astype(min_times.dtype)
        diff = prs_began - work_began
        zero = diff.dtype.type(0)
        diff[diff < zero] = zero
        statuses = facts[Status.category_name.name].values
        diff[(statuses != Status.CATEGORY_IN_PROGRESS) & (statuses != Status.CATEGORY_DONE)] = None
        focus_mask = (min_times[:, None] <= prs_began) & (prs_began < max_times[:, None])
        result[:] = diff
        result[~focus_mask] = None
        return result


@register_metric(JIRAMetricID.JIRA_BACKLOG_TIME)
class BacklogTimeCalculator(AverageMetricCalculator[timedelta]):
    """
    Issue Backlog Time calculator.

    The time between when the issue was created and transitioned to "In Progress".
    """

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **_,
    ) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, self.dtype)
        created = facts[Issue.created.name].values.astype(min_times.dtype)
        work_began = facts[AthenianIssue.work_began.name].values.astype(min_times.dtype)
        diff = work_began - created
        statuses = facts[Status.category_name.name].values
        diff[(statuses != Status.CATEGORY_IN_PROGRESS) & (statuses != Status.CATEGORY_DONE)] = None
        focus_mask = (min_times[:, None] <= work_began) & (work_began < max_times[:, None])
        result[:] = diff
        result[~focus_mask] = None
        return result
