from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Type

import numpy as np
import pandas as pd

from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    BinnedEnsemblesCalculator, FlowRatioCalculator, MetricCalculator, MetricCalculatorEnsemble, \
    SumMetricCalculator
from athenian.api.controllers.miners.jira.issue import ISSUE_PRS_BEGAN, ISSUE_PRS_RELEASED
from athenian.api.models.metadata.jira import AthenianIssue, Issue
from athenian.api.models.web import JIRAMetricID
from athenian.api.tracing import sentry_span

metric_calculators: Dict[str, Type[MetricCalculator]] = {}


def register_metric(name: str):
    """Keep track of the release metric calculators."""
    assert isinstance(name, str)

    def register_with_name(cls: Type[MetricCalculator]):
        metric_calculators[name] = cls
        return cls

    return register_with_name


class JIRAMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for JIRA issues."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of JIRAMetricCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=metric_calculators)


class JIRABinnedMetricCalculator(BinnedEnsemblesCalculator[Metric]):
    """
    BinnedMetricCalculator adapted for JIRA issues.

    We've got a completely custom __call__ method to avoid the redundant complexity of the parent.
    """

    ensemble_class = JIRAMetricCalculatorEnsemble

    def __init__(self,
                 metrics: Sequence[str],
                 quantiles: Sequence[float],
                 **kwargs):
        """
        Initialize a new instance of `JIRABinnedMetricCalculator`.

        :param metrics: Sequence of metric names to calculate in each bin.
        :param quantiles: Pair of quantiles, common for each metric.
        """
        super().__init__([metrics], quantiles, **kwargs)

    @sentry_span
    def __call__(self,
                 items: pd.DataFrame,
                 time_intervals: Sequence[Sequence[datetime]],
                 groups: Sequence[np.ndarray],
                 ) -> List[List[List[List[Metric]]]]:
        """
        Calculate the binned aggregations on a series of mined issues.

        :param items: pd.DataFrame with the fetched issues data.
        :param time_intervals: Time interval borders in UTC. Each interval spans \
                               `[time_intervals[i], time_intervals[i + 1]]`, the ending \
                               not included.
        :param groups: Various issue groups, the metrics will be calculated independently \
                       for each group.
        :return: groups x time intervals primary x time intervals secondary x metrics.
        """
        min_times, max_times, ts_index_map = self._make_min_max_times(time_intervals)
        assert len(self.ensembles) == 1
        ensemble = self.ensembles[0]
        ensemble(items, min_times, max_times, groups)
        values_dict = ensemble.values()
        result = [[[[None] * len(self.metrics[0])
                    for _ in range(len(ts) - 1)]
                   for ts in time_intervals]
                  for _ in groups]
        for mix, metric in enumerate(self.metrics[0]):
            for gi in range(len(groups)):
                for (primary, secondary), value in zip(ts_index_map, values_dict[metric][gi]):
                    result[gi][primary][secondary][mix] = value
        return result


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
        released = facts[ISSUE_PRS_RELEASED].values.astype(min_times.dtype)[prs_began == prs_began]
        resolved[prs_began == prs_began][released != released] = np.datetime64("nat")
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
        released = facts[ISSUE_PRS_RELEASED].values.astype(min_times.dtype)[prs_began == prs_began]
        resolved[prs_began == prs_began][released != released] = np.datetime64("nat")
        not_resolved = resolved != resolved
        resolved_later = resolved >= max_times[:, None]
        created_earlier = created < max_times[:, None]
        result[(resolved_later | not_resolved) & created_earlier] = 1
        return result


@register_metric(JIRAMetricID.JIRA_FLOW_RATIO)
class IssueFlowRatioCalculator(FlowRatioCalculator):
    """Calculate JIRA issues flow ratio = raised / resolved."""

    deps = (OpenCounter, ResolvedCounter)


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
        life_times = np.maximum(released, resolved) - np.minimum(created, prs_began)
        nat = np.datetime64("nat")
        life_times[released != released] = nat
        life_times[resolved != resolved] = nat
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
        work_began = facts[AthenianIssue.work_began.key].values
        prs_began = facts[ISSUE_PRS_BEGAN].values.astype(min_times.dtype)
        resolved = facts[Issue.resolved.key].values.astype(min_times.dtype)
        released = facts[ISSUE_PRS_RELEASED].values.astype(min_times.dtype)
        focus_mask = (min_times[:, None] <= resolved) & (resolved < max_times[:, None])
        lead_times = np.maximum(released, resolved) - np.minimum(work_began, prs_began)
        nat = np.datetime64("nat")
        lead_times[released != released] = nat
        lead_times[resolved != resolved] = nat
        unmapped_mask = prs_began != prs_began
        lead_times[unmapped_mask] = resolved[unmapped_mask] - work_began[unmapped_mask]
        empty_lead_time_mask = lead_times != lead_times
        lead_times = lead_times.astype(self.dtype).view(int)
        result[:] = lead_times
        result[:, empty_lead_time_mask] = None
        result[~focus_mask] = None
        return result
