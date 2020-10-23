from datetime import datetime
from typing import Dict, List, Sequence, Type

import numpy as np
import pandas as pd

from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.features.metric_calculator import BinnedEnsemblesCalculator, \
    MetricCalculator, MetricCalculatorEnsemble, SumMetricCalculator
from athenian.api.models.metadata.jira import Issue
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
                 ) -> List[List[List[Metric]]]:
        """
        Calculate the binned aggregations on a series of mined issues.

        :param items: pd.DataFrame with the fetched issues data.
        :param time_intervals: Time interval borders in UTC. Each interval spans \
                               `[time_intervals[i], time_intervals[i + 1]]`, the ending \
                               not included.
        :return: time intervals primary x time intervals secondary x metrics.
        """
        min_times, max_times, ts_index_map = self._make_min_max_times(time_intervals)
        assert len(self.ensembles) == 1
        ensemble = self.ensembles[0]
        ensemble(items, min_times, max_times, [np.arange(len(items))])
        values_dict = ensemble.values()
        result = [[[None] * len(self.metrics[0])
                   for _ in range(len(ts) - 1)]
                  for ts in time_intervals]
        for mix, metric in enumerate(self.metrics[0]):
            for (primary, secondary), value in zip(ts_index_map, values_dict[metric][0]):
                result[primary][secondary][mix] = value
        return result


@register_metric(JIRAMetricID.JIRA_BUG_RAISED)
class RaisedCounter(SumMetricCalculator[int]):
    """Number of created issues metric."""

    may_have_negative_values = False
    dtype = int

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        created = facts[Issue.created.key].values
        is_bug = facts[Issue.type.key].str.lower().values == "bug"
        result[(min_times[:, None] <= created) & (created < max_times[:, None]) & is_bug] = 1
        return result


@register_metric(JIRAMetricID.JIRA_BUG_RESOLVED)
class ResolvedCounter(SumMetricCalculator[int]):
    """Number of resolved issues metric."""

    may_have_negative_values = False
    dtype = int

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        resolved = facts[Issue.resolved.key].values
        is_bug = facts[Issue.type.key].str.lower().values == "bug"
        result[(min_times[:, None] <= resolved) & (resolved < max_times[:, None]) & is_bug] = 1
        return result


@register_metric(JIRAMetricID.JIRA_BUG_OPEN)
class OpenCounter(SumMetricCalculator[int]):
    """Number of created issues metric."""

    may_have_negative_values = False
    dtype = int

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        created = facts[Issue.created.key].values
        resolved = facts[Issue.resolved.key].values
        is_bug = facts[Issue.type.key].str.lower().values == "bug"
        not_resolved = resolved != resolved
        resolved_later = resolved >= max_times[:, None]
        created_earlier = created < max_times[:, None]
        result[is_bug & (resolved_later | not_resolved) & created_earlier] = 1
        return result
