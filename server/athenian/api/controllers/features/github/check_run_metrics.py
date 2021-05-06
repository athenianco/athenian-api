from typing import Callable, Dict, List, Sequence, Tuple, Type

import numpy as np
import pandas as pd

from athenian.api.controllers.features.metric_calculator import BinnedHistogramCalculator, \
    BinnedMetricCalculator, HistogramCalculatorEnsemble, make_register_metric, MetricCalculator, \
    MetricCalculatorEnsemble
from athenian.api.models.metadata.github import CheckRun

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
        Sequence[int]]:
    """
    Split check runs by parent check suite size.

    :return: Function to return the groups + check suite size frequencies.
    """
    suites = df[CheckRun.check_suite_node_id.key].values.astype("S")
    unique_suites, run_counts = np.unique(suites, return_counts=True)
    suite_blocks = np.array(np.split(np.argsort(suites), np.cumsum(run_counts[:-1])))
    suite_size_counts, back_indexes, group_counts = np.unique(
        run_counts, return_inverse=True, return_counts=True)
    run_counts_order = np.argsort(back_indexes)
    ordered_indexes = np.concatenate(suite_blocks[run_counts_order])
    groups = np.split(ordered_indexes, np.cumsum(group_counts[:-1]))

    def group_check_runs_by_check_runs_count(_) -> List[np.ndarray]:
        return groups

    return group_check_runs_by_check_runs_count, suite_size_counts
