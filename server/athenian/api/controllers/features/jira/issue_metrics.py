from datetime import timedelta
from typing import Dict, Sequence, Type

import numpy as np
import pandas as pd

from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    BinnedHistogramCalculator, BinnedMetricCalculator, HistogramCalculator, \
    HistogramCalculatorEnsemble, make_register_metric, MetricCalculator, \
    MetricCalculatorEnsemble, RatioCalculator, SumMetricCalculator
from athenian.api.controllers.miners.jira.issue import ISSUE_PRS_BEGAN, ISSUE_PRS_RELEASED
from athenian.api.models.metadata.jira import AthenianIssue, Issue
from athenian.api.models.web import JIRAMetricID


metric_calculators: Dict[str, Type[MetricCalculator]] = {}
histogram_calculators: Dict[str, Type[HistogramCalculator]] = {}
register_metric = make_register_metric(metric_calculators, histogram_calculators)


class JIRAMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for JIRA issues."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of JIRAMetricCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=metric_calculators)


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


@register_metric(JIRAMetricID.JIRA_RAISED)
class RaisedCounter(SumMetricCalculator[int]):
    """Number of created issues metric."""

    dtype = int

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        created = facts[Issue.created.key].values
        result[(min_times[:, None] <= created) & (created < max_times[:, None])] = 1
        return result


@register_metric(JIRAMetricID.JIRA_RESOLVED)
class ResolvedCounter(SumMetricCalculator[int]):
    """Number of resolved issues metric."""

    dtype = int

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        resolved = facts[AthenianIssue.resolved.key].values.astype(min_times.dtype)
        prs_began = facts[ISSUE_PRS_BEGAN].values.astype(min_times.dtype)
        have_prs_mask = prs_began == prs_began
        released = facts[ISSUE_PRS_RELEASED].values.astype(min_times.dtype)[have_prs_mask]
        resolved[have_prs_mask][released != released] = np.datetime64("nat")
        result[(min_times[:, None] <= resolved) & (resolved < max_times[:, None])] = 1
        return result


@register_metric(JIRAMetricID.JIRA_OPEN)
class OpenCounter(SumMetricCalculator[int]):
    """Number of created issues metric."""

    dtype = int

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        created = facts[Issue.created.key].values
        resolved = facts[AthenianIssue.resolved.key].values.astype(min_times.dtype)
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
    dtype = "timedelta64[s]"

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        created = facts[Issue.created.key].values
        prs_began = facts[ISSUE_PRS_BEGAN].values.astype(min_times.dtype)
        resolved = facts[Issue.resolved.key].values.astype(min_times.dtype)
        released = facts[ISSUE_PRS_RELEASED].values.astype(min_times.dtype)
        focus_mask = (min_times[:, None] <= resolved) & (resolved < max_times[:, None])
        life_times = np.maximum(released, resolved) - np.fmin(created, prs_began)
        unmapped_mask = prs_began != prs_began
        life_times[unmapped_mask] = resolved[unmapped_mask] - created[unmapped_mask]
        empty_life_time_mask = life_times != life_times
        life_times = life_times.astype(self.dtype).view(int)
        result[:] = life_times
        result[:, empty_life_time_mask] = None
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
    dtype = "timedelta64[s]"

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        work_began = facts[AthenianIssue.work_began.key].values.astype(min_times.dtype)
        prs_began = facts[ISSUE_PRS_BEGAN].values.astype(min_times.dtype)
        resolved = facts[Issue.resolved.key].values.astype(min_times.dtype)
        released = facts[ISSUE_PRS_RELEASED].values.astype(min_times.dtype)
        focus_mask = (min_times[:, None] <= resolved) & (resolved < max_times[:, None])
        lead_times = np.maximum(released, resolved) - np.fmin(work_began, prs_began)
        unmapped_mask = prs_began != prs_began
        lead_times[unmapped_mask] = resolved[unmapped_mask] - work_began[unmapped_mask]
        empty_lead_time_mask = lead_times != lead_times
        lead_times = lead_times.astype(self.dtype).view(int)
        result[:] = lead_times
        result[:, empty_lead_time_mask] = None
        result[~focus_mask] = None
        return result


@register_metric(JIRAMetricID.JIRA_ACKNOWLEDGE_TIME)
class AcknowledgeTimeCalculator(AverageMetricCalculator[timedelta]):
    """
    Issue Acknowledge Time calculator.

    Acknowledge time is the time it takes for the work to actually being after the issue is \
    created.

    * If the work began before the creation time, the acknowledge time is 0.
    """

    may_have_negative_values = False
    dtype = "timedelta64[s]"

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        work_began = facts[AthenianIssue.work_began.key].values.astype(min_times.dtype)
        prs_began = facts[ISSUE_PRS_BEGAN].values.astype(min_times.dtype)
        acknowledged = np.fmin(work_began, prs_began)
        created = facts[Issue.created.key].values.astype(min_times.dtype)
        ack_times = acknowledged - created
        ack_times = np.maximum(ack_times, ack_times.dtype.type(0))
        focus_mask = (min_times[:, None] <= acknowledged) & (acknowledged < max_times[:, None])
        result[:] = ack_times.astype(self.dtype).view(int)
        result[~focus_mask] = None
        return result
