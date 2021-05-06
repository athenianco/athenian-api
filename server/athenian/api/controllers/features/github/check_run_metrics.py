from typing import Callable, Dict, List, Sequence, Tuple, Type

import numpy as np
import pandas as pd

from athenian.api.controllers.features.metric_calculator import BinnedHistogramCalculator, \
    BinnedMetricCalculator, HistogramCalculatorEnsemble, make_register_metric, MetricCalculator, \
    MetricCalculatorEnsemble, SumMetricCalculator
from athenian.api.models.metadata.github import CheckRun
from athenian.api.models.web import CodeCheckMetricID

metric_calculators: Dict[str, Type[MetricCalculator]] = {}
histogram_calculators: Dict[str, Type[MetricCalculator]] = {}
register_metric = make_register_metric(metric_calculators, None)


class CheckRunMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for CI check runs."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of CheckRunMetricCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=metric_calculators)


class CheckRunHistogramCalculatorEnsemble(HistogramCalculatorEnsemble):
    """HistogramCalculatorEnsemble adapted for CI check runs."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of CheckRunHistogramCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=histogram_calculators)


class CheckRunBinnedMetricCalculator(BinnedMetricCalculator):
    """BinnedMetricCalculator adapted for CI check runs."""

    ensemble_class = CheckRunMetricCalculatorEnsemble


class CheckRunBinnedHistogramCalculator(BinnedHistogramCalculator):
    """BinnedHistogramCalculator adapted for CI check runs."""

    ensemble_class = CheckRunHistogramCalculatorEnsemble


def group_check_runs_by_commit_authors(commit_authors: List[List[str]],
                                       df: pd.DataFrame,
                                       ) -> List[np.ndarray]:
    """Triage check runs by their corresponding commit authors."""
    if not commit_authors or df.empty:
        return [np.arange(len(df))]
    indexes = []
    for group in commit_authors:
        group = np.unique(group).astype("S")
        commit_authors = df[CheckRun.author_login.key].values.astype("S")
        included_indexes = np.nonzero(np.in1d(commit_authors, group))[0]
        indexes.append(included_indexes)
    return indexes


def make_check_runs_count_grouper(df: pd.DataFrame) -> Tuple[
        Callable[[pd.DataFrame], List[np.ndarray]],
        np.ndarray,
        Sequence[int]]:
    """
    Split check runs by parent check suite size.

    :return: 1. Function to return the groups. \
             2. Check suite node IDs column. \
             3. Check suite sizes.
    """
    suites = df[CheckRun.check_suite_node_id.key].values.astype("S")
    unique_suites, run_counts = np.unique(suites, return_counts=True)
    suite_blocks = np.array(np.split(np.argsort(suites), np.cumsum(run_counts)[:-1]))
    unique_run_counts, back_indexes, group_counts = np.unique(
        run_counts, return_inverse=True, return_counts=True)
    run_counts_order = np.argsort(back_indexes)
    ordered_indexes = np.concatenate(suite_blocks[run_counts_order])
    groups = np.split(ordered_indexes, np.cumsum(group_counts * unique_run_counts)[:-1])

    def group_check_runs_by_check_runs_count(_) -> List[np.ndarray]:
        return groups

    return group_check_runs_by_check_runs_count, suites, unique_run_counts


@register_metric(CodeCheckMetricID.SUITES_COUNT)
class SuitesCounter(SumMetricCalculator[int]):
    """Number of executed check suites metric."""

    dtype = int

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        started = facts[CheckRun.started_at.key].values.astype(min_times.dtype)
        _, first_encounters = np.unique(
            facts[CheckRun.check_suite_node_id.key].values.astype("S"), return_index=True)
        mask = np.zeros_like(started, dtype=bool)
        mask[first_encounters] = True
        started[~mask] = None
        result[(min_times[:, None] <= started) & (started < max_times[:, None])] = 1
        return result


class SuitesInStatusCounter(SumMetricCalculator[int]):
    """Number of executed check suites metric with the specified `statuses`."""

    dtype = int
    statuses = {}

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), None, object)
        started = facts[CheckRun.started_at.key].values.astype(min_times.dtype)
        statuses = facts[CheckRun.check_suite_status.key].values.astype("S")
        conclusions = facts[CheckRun.check_suite_conclusion.key].values.astype("S")
        relevant = np.zeros_like(started, dtype=bool)
        for status, status_conclusions in self.statuses.items():
            status_mask = statuses == status
            if status_conclusions:
                mask = np.zeros_like(status_mask)
                for sc in status_conclusions:
                    mask |= conclusions == sc
                mask &= status_mask
            else:
                mask = status_mask
            relevant |= mask
        _, first_encounters = np.unique(
            facts[CheckRun.check_suite_node_id.key].values.astype("S"), return_index=True)
        mask = np.zeros_like(relevant)
        mask[first_encounters] = True
        relevant[~mask] = False
        started[~relevant] = None
        result[(min_times[:, None] <= started) & (started < max_times[:, None])] = 1
        return result


@register_metric(CodeCheckMetricID.SUCCESSFUL_SUITES_COUNT)
class SuccessfulSuitesCounter(SuitesInStatusCounter):
    """Number of successfully executed check suites metric."""

    statuses = {
        b"COMPLETED": [b"SUCCESS"],
        b"SUCCESS": [],
    }


@register_metric(CodeCheckMetricID.FAILED_SUITES_COUNT)
class FailedSuitesCounter(SuitesInStatusCounter):
    """Number of failed check suites metric."""

    statuses = {
        b"COMPLETED": [b"FAILURE", b"STALE"],
        b"FAILURE": [],
    }


@register_metric(CodeCheckMetricID.CANCELLED_SUITES_COUNT)
class CancelledSuitesCounter(SuitesInStatusCounter):
    """Number of cancelled check suites metric."""

    statuses = {
        b"COMPLETED": [b"CANCELLED"],
    }
