from typing import Dict, List, Sequence, Type

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
