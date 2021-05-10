from datetime import timedelta
from typing import Callable, Dict, List, Sequence, Tuple, Type
import warnings

import numpy as np
import pandas as pd

from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    BinnedHistogramCalculator, BinnedMetricCalculator, HistogramCalculatorEnsemble, \
    make_register_metric, MetricCalculator, MetricCalculatorEnsemble, RatioCalculator, \
    SumMetricCalculator
from athenian.api.controllers.miners.github.check_run import check_suite_started_column, \
    pull_request_closed_column, pull_request_merged_column, pull_request_started_column
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
        result = np.zeros((len(min_times), len(facts)), int)
        started = facts[check_suite_started_column].values.astype(min_times.dtype)
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
        started = facts[check_suite_started_column].values.astype(min_times.dtype)
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
        result = np.zeros((len(min_times), len(facts)), int)
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
        b"COMPLETED": [b"CANCELLED", b"SKIPPED"],
    }


@register_metric(CodeCheckMetricID.SUCCESS_RATIO)
class SuiteSuccessRatioCalculator(RatioCalculator):
    """Ratio of successful check suites divided by the overall count."""

    deps = (SuccessfulSuitesCounter, SuitesCounter)


@register_metric(CodeCheckMetricID.SUITE_TIME)
class SuiteTimeCalculatorAnalysis(MetricCalculator[None]):
    """Measure elapsed time and size of each check suite."""

    may_have_negative_values = False
    dtype = np.dtype([("elapsed", "timedelta64[s]"), ("size", int)])
    has_nan = True

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        unique_suites, first_encounters, inverse_indexes, run_counts = np.unique(
            facts[CheckRun.check_suite_node_id.key].values.astype("S"),
            return_index=True, return_inverse=True, return_counts=True)
        statuses = facts[CheckRun.check_suite_status.key].values[first_encounters].astype("S")
        completed = statuses == b"COMPLETED"
        conclusions = facts[CheckRun.check_suite_conclusion.key].values[
            first_encounters[completed]
        ].astype("S")
        sensibly_completed = np.nonzero(completed)[0][
            np.in1d(conclusions, [b"SUCCESS", b"FAILURE", b"STALE"])]
        # first_encounters[sensibly_completed] gives the indexes of the completed suites
        first_encounters = first_encounters[sensibly_completed]

        # measure elapsed time for each suite size group
        suite_blocks = np.array(np.split(np.argsort(inverse_indexes), np.cumsum(run_counts)[:-1]))
        unique_run_counts, back_indexes, group_counts = np.unique(
            run_counts, return_inverse=True, return_counts=True)
        run_counts_order = np.argsort(back_indexes)
        ordered_indexes = np.concatenate(suite_blocks[run_counts_order])
        groups = np.split(ordered_indexes, np.cumsum(group_counts * unique_run_counts)[:-1])
        run_finished = facts[CheckRun.completed_at.key].values.astype(min_times.dtype)
        suite_started_col = facts[check_suite_started_column].values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered")
            suite_finished = np.concatenate([
                np.nanmax(run_finished[group].reshape(-1, unique_run_count), axis=1)
                for group, unique_run_count in zip(groups, unique_run_counts)
            ])
            suite_started = np.concatenate([
                suite_started_col[group].reshape(-1, unique_run_count)[:, 0]
                for group, unique_run_count in zip(groups, unique_run_counts)
            ])
        elapsed = suite_finished - suite_started
        # if the check takes very little time, the timestamps may break
        elapsed[elapsed < np.array([0], dtype=elapsed.dtype)] = 0
        # reorder the sequence to match unique_suites
        suite_order = np.argsort(run_counts_order)
        structs = np.zeros_like(suite_order, dtype=self.dtype)
        structs["elapsed"] = elapsed[suite_order]
        structs["size"] = np.repeat(unique_run_counts, group_counts)[suite_order]
        suite_started = suite_started[suite_order]

        result = np.full((len(min_times), len(facts)),
                         np.array([(np.timedelta64("NaT"), -1)], dtype=self.dtype))
        time_relevant_suite_mask = \
            (min_times[:, None] <= suite_started) & (suite_started < max_times[:, None])
        result_structs = np.repeat(structs[sensibly_completed][None, :], len(min_times), axis=0)
        mask = ~time_relevant_suite_mask[:, sensibly_completed]
        result_structs[mask]["elapsed"] = np.timedelta64("NaT")
        result_structs[mask]["size"] = -1
        result[:, first_encounters] = result_structs
        return result

    def _value(self, samples: np.ndarray) -> Metric[None]:
        return Metric(False, None, None, None)


@register_metric(CodeCheckMetricID.SUITE_TIME)
class SuiteTimeCalculator(AverageMetricCalculator[timedelta]):
    """Average check suite execution time metric."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"
    deps = (SuiteTimeCalculatorAnalysis,)

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        return self._calcs[0].peek.copy()["elapsed"].astype(object)


@register_metric(CodeCheckMetricID.ROBUST_SUCCESS_RATIO)
class RobustSuiteTimeCalculator(AverageMetricCalculator[timedelta]):
    """Average check suite execution time metric, sustainable version."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        structs = self._calcs[0].peek
        _ = structs
        raise NotImplementedError


@register_metric(CodeCheckMetricID.SUITES_PER_PR)
class SuitesPerPRCounter(AverageMetricCalculator[float]):
    """Average number of executed check suites per pull request metric."""

    may_have_negative_values = False
    dtype = float

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        unique_suites, first_suite_encounters = np.unique(
            facts[CheckRun.check_suite_node_id.key].values.astype("S"), return_index=True)
        """
        # only completed
        completed = np.in1d(
            facts[CheckRun.check_suite_status.key].values[first_suite_encounters].astype("S"),
            [b"COMPLETED", b"SUCCESS", b"FAILURE"])
        first_suite_encounters = first_suite_encounters[completed]
        """
        pull_requests = facts[CheckRun.pull_request_node_id.key].values.astype("S")
        unique_prs, first_pr_encounters = np.unique(pull_requests, return_index=True)
        none_mask = unique_prs == b"None"
        none_pos = np.nonzero(none_mask)[0]
        not_none_mask = ~none_mask
        del none_mask
        if len(none_pos):
            first_pr_encounters = first_pr_encounters[not_none_mask]
        unique_prs_alt, pr_suite_counts = np.unique(
            pull_requests[first_suite_encounters], return_counts=True)
        assert unique_prs.shape == unique_prs_alt.shape, \
            f"invalid PR deduplication: {len(unique_prs)} vs. {len(unique_prs_alt)}"
        del unique_prs_alt
        del unique_prs
        if len(none_pos):
            pr_suite_counts = pr_suite_counts[not_none_mask]
        mask_pr_times = (
            (facts[pull_request_started_column].values < max_times[:, None])
            &
            (facts[pull_request_closed_column].values >= min_times[:, None])
        )
        result = np.zeros((len(min_times), len(facts)), dtype=np.float32)
        result[:, first_pr_encounters] = pr_suite_counts
        result[~mask_pr_times] = None
        return result


@register_metric(CodeCheckMetricID.SUITE_TIME_PER_PR)
class SuiteTimePerPRCalculator(AverageMetricCalculator[timedelta]):
    """Average check suite execution in PRs time metric."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"
    deps = (SuiteTimeCalculator,)

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        pull_requests = facts[CheckRun.pull_request_node_id.key].values.astype("S")
        no_prs_mask = pull_requests == b"None"
        result = self._calcs[0].peek.copy()
        result[:, no_prs_mask] = None
        return result


@register_metric(CodeCheckMetricID.PRS_WITH_CHECKS_COUNT)
class PRsWithChecksCounter(SumMetricCalculator[int]):
    """Number of PRs with executed check suites."""

    dtype = int
    deps = (SuitesPerPRCounter,)

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        result = self._calcs[0].peek.copy()
        result[result > 0] = 1
        return result


@register_metric(CodeCheckMetricID.FLAKY_COMMIT_CHECKS_COUNT)
class FlakyCommitChecksCounter(SumMetricCalculator[int]):
    """Number of commits with both successful and failed check suites."""

    dtype = int
    deps = (SuccessfulSuitesCounter, FailedSuitesCounter)

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        statuses = facts[CheckRun.status.key].values.astype("S")
        conclusions = facts[CheckRun.conclusion.key].values.astype("S")
        completed = statuses == b"COMPLETED"
        success_mask = (
            (completed & ((conclusions == b"SUCCESS") | (conclusions == b"NEUTRAL"))) |
            (statuses == b"SUCCESS")
        )
        failure_mask = (
            (completed & ((conclusions == b"FAILURE") | (conclusions == b"STALE"))) |
            (statuses == b"FAILURE") | (statuses == b"ERROR")
        )

        commits = facts[CheckRun.commit_node_id.key].values.astype("S")
        check_run_names = facts[CheckRun.name.key].values.astype("S")
        commits_with_names = np.char.add(commits, check_run_names)
        _, first_encounters, unique_map = np.unique(
            commits_with_names, return_inverse=True, return_index=True)
        unique_flaky_indexes = np.intersect1d(unique_map[success_mask], unique_map[failure_mask])
        first_flaky_indexes = first_encounters[unique_flaky_indexes]

        # some commits may count several times => reduce
        flaky_commits = np.unique(commits[first_flaky_indexes])
        unique_commits, first_encounters = np.unique(commits, return_index=True)
        flaky_mask = np.in1d(unique_commits, flaky_commits, assume_unique=True)
        first_flaky_commits = first_encounters[flaky_mask]

        started = facts[check_suite_started_column].values.astype(min_times.dtype)
        result = np.zeros((len(min_times), len(facts)), int)
        mask = np.zeros(len(facts), bool)
        mask[first_flaky_commits] = 1
        started[~mask] = None
        mask = (min_times[:, None] <= started) & (started < max_times[:, None])
        result[mask] = 1
        return result


@register_metric(CodeCheckMetricID.PRS_MERGED_WITH_FAILED_CHECKS_COUNT)
class MergedPRsWithFailedChecksCounter(SumMetricCalculator[int]):
    """Count how many PRs were merged despite failing checks."""

    dtype = int

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        df = facts[[
            CheckRun.pull_request_node_id.key, CheckRun.name.key, CheckRun.started_at.key,
            CheckRun.status.key, CheckRun.conclusion.key, pull_request_merged_column]]
        df = df.sort_values(CheckRun.started_at.key, ascending=False)  # no inplace=True, yes
        pull_requests = df[CheckRun.pull_request_node_id.key].values.astype("S")
        names = df[CheckRun.name.key].values.astype("S")
        joint = np.char.add(pull_requests, names)
        _, first_encounters = np.unique(joint, return_index=True)
        statuses = df[CheckRun.status.key].values.astype("S")
        conclusions = df[CheckRun.conclusion.key].values.astype("S")
        failure_mask = np.zeros_like(statuses, dtype=bool)
        failure_mask[first_encounters] = True
        failure_mask &= (
            ((statuses == b"COMPLETED") & (
                (conclusions == b"FAILURE") | (conclusions == b"STALE"))
             ) | (statuses == b"FAILURE") | (statuses == b"ERROR")
        ) & (pull_requests != b"None") & df[pull_request_merged_column].values
        failing_pull_requests = pull_requests[failure_mask]
        _, failing_indexes = np.unique(failing_pull_requests, return_index=True)
        failing_indexes = np.nonzero(failure_mask)[0][failing_indexes]
        failing_indexes = df.index.values[failing_indexes]

        mask_pr_times = (
            (facts[pull_request_started_column].values < max_times[:, None])
            &
            (facts[pull_request_closed_column].values >= min_times[:, None])
        )
        result = np.zeros((len(min_times), len(facts)), dtype=int)
        result[:, failing_indexes] = 1
        result[~mask_pr_times] = 0
        return result


class MergedPRsCounter(SumMetricCalculator[int]):
    """Count how many PRs were merged with checks."""

    dtype = int

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        pull_requests = facts[CheckRun.pull_request_node_id.key].values.astype("S")
        merged = facts[pull_request_merged_column].values
        pull_requests[~merged] = b"None"
        unique_prs, first_encounters = np.unique(pull_requests, return_index=True)
        first_encounters = first_encounters[unique_prs != b"None"]
        mask_pr_times = (
            (facts[pull_request_started_column].values < max_times[:, None])
            &
            (facts[pull_request_closed_column].values >= min_times[:, None])
        )
        result = np.zeros((len(min_times), len(facts)), dtype=int)
        result[:, first_encounters] = 1
        result[~mask_pr_times] = 0
        return result


@register_metric(CodeCheckMetricID.PRS_MERGED_WITH_FAILED_CHECKS_RATIO)
class MergedPRsWithFailedChecksRatioCalculator(RatioCalculator):
    """Calculate the ratio of PRs merged with failing checks to all merged PRs with checks."""

    deps = (MergedPRsWithFailedChecksCounter, MergedPRsCounter)
