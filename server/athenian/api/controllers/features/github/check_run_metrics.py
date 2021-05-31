from datetime import timedelta
from typing import Callable, Dict, List, Sequence, Tuple, Type
import warnings

import numpy as np
import pandas as pd

from athenian.api.controllers.features.github.check_run_metrics_accelerated import \
    calculate_interval_intersections
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    BinnedHistogramCalculator, BinnedMetricCalculator, HistogramCalculatorEnsemble, \
    make_register_metric, MaxMetricCalculator, MetricCalculator, MetricCalculatorEnsemble, \
    RatioCalculator, SumMetricCalculator
from athenian.api.controllers.miners.github.check_run import check_suite_started_column, \
    pull_request_closed_column, pull_request_merged_column, pull_request_started_column
from athenian.api.models.metadata.github import CheckRun
from athenian.api.models.web import CodeCheckMetricID

metric_calculators: Dict[str, Type[MetricCalculator]] = {}
histogram_calculators: Dict[str, Type[MetricCalculator]] = {}
register_metric = make_register_metric(metric_calculators, histogram_calculators)


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


def group_check_runs_by_pushers(pushers: List[List[str]],
                                df: pd.DataFrame,
                                ) -> List[np.ndarray]:
    """Triage check runs by their corresponding commit authors."""
    if not pushers or df.empty:
        return [np.arange(len(df))]
    indexes = []
    for group in pushers:
        group = np.unique(group).astype("S")
        pushers = df[CheckRun.author_login.key].values.astype("S")
        included_indexes = np.nonzero(np.in1d(pushers, group))[0]
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
        b"ERROR": [],
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
        suite_blocks = np.array(np.split(np.argsort(inverse_indexes), np.cumsum(run_counts)[:-1]),
                                dtype=object)
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
        # reorder the sequence to match unique_suites
        suite_order = np.argsort(run_counts_order)
        structs = np.zeros_like(suite_order, dtype=self.dtype)
        structs["elapsed"] = elapsed[suite_order]
        structs["size"] = np.repeat(unique_run_counts, group_counts)[suite_order]
        suite_started = suite_started[suite_order]

        result = np.full((len(min_times), len(facts)),
                         np.array([(np.timedelta64("NaT"), 0)], dtype=self.dtype))
        time_relevant_suite_mask = \
            (min_times[:, None] <= suite_started) & (suite_started < max_times[:, None])
        result_structs = np.repeat(structs[sensibly_completed][None, :], len(min_times), axis=0)
        mask = ~time_relevant_suite_mask[:, sensibly_completed]
        result_structs["elapsed"][mask] = np.timedelta64("NaT")
        result_structs["size"][mask] = 0
        result[:, first_encounters] = result_structs
        return result

    def _value(self, samples: np.ndarray) -> Metric[None]:
        return Metric(False, None, None, None)


@register_metric(CodeCheckMetricID.SUITE_TIME)
class SuiteTimeCalculator(AverageMetricCalculator[timedelta]):
    """Average check suite execution time metric."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"
    has_nan = True
    deps = (SuiteTimeCalculatorAnalysis,)

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        return self._calcs[0].peek["elapsed"].astype(self.dtype, copy=False)


@register_metric(CodeCheckMetricID.ROBUST_SUITE_TIME)
class RobustSuiteTimeCalculator(MetricCalculator[timedelta]):
    """Average check suite execution time metric, sustainable version."""

    dtype = "timedelta64[s]"
    deps = (SuiteTimeCalculatorAnalysis,)

    def __call__(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 groups: Sequence[Sequence[int]],
                 **kwargs) -> None:
        """Completely ignore the default boilerplate and calculate metrics from scratch."""
        structs = self._calcs[0].peek
        meaningful = np.nonzero(np.bitwise_or.reduce(structs["size"]) > 0)[0]
        elapsed = structs["elapsed"]
        sizes = structs["size"].astype("S")
        repos = facts[CheckRun.repository_node_id.key].values.astype("S")
        repos_sizes = np.char.add(repos, np.char.add(b"|", sizes))
        self._metrics = metrics = []
        for group in groups:
            effective = np.intersect1d(group, meaningful, assume_unique=True)
            group_repos_sizes = repos_sizes[:, effective]
            group_elapsed = elapsed[:, effective]
            if (self._quantiles[0] != 0 or self._quantiles[1] != 1) and len(group_elapsed) > 0:
                not_nat_ts, not_nat_facts = np.nonzero(group_elapsed == group_elapsed)
                if len(not_nat_facts):
                    not_nat_facts_unique, first_encounters = np.unique(
                        not_nat_facts, return_index=True)
                    quantiles = self._calc_quantile_cut_values(
                        group_elapsed[not_nat_ts[first_encounters], not_nat_facts_unique])
                    if self._quantiles[0] != 0:
                        group_repos_sizes[group_elapsed < quantiles[0]] = b"|0"
                    if self._quantiles[1] != 1:
                        group_repos_sizes[group_elapsed > quantiles[1]] = b"|0"
            vocabulary, mapped_indexes = np.unique(group_repos_sizes, return_inverse=True)
            existing_vocabulary_indexes = np.nonzero(~np.char.endswith(vocabulary, b"|0"))[0]
            masks_by_reposize = (
                mapped_indexes[:, np.newaxis] == existing_vocabulary_indexes
            ).T.reshape((len(existing_vocabulary_indexes), *group_repos_sizes.shape))
            # we don't call mean() because there may be empty slices
            sums = np.sum(np.broadcast_to(group_elapsed[None, :],
                                          (len(masks_by_reposize), *group_elapsed.shape)),
                          axis=-1, where=masks_by_reposize)
            counts = np.sum(masks_by_reposize, axis=-1)
            # backfill
            if (missing := counts == 0).any():
                existing_reposizes, existing_ts = np.array(np.nonzero(np.flip(~missing, axis=1)))
                _, existing_reposizes_counts = np.unique(existing_reposizes, return_counts=True)
                existing_borders = np.cumsum(existing_reposizes_counts)[:-1]
                saturated_existing_ts = existing_ts.copy()
                ts_len = counts.shape[-1]
                saturated_existing_ts[existing_borders - 1] = ts_len
                saturated_existing_ts[-1] = ts_len
                offsets = np.diff(np.insert(saturated_existing_ts, existing_borders, 0), prepend=0)
                offsets = np.delete(offsets, existing_borders + np.arange(len(existing_borders)))
                reposize_indexes = np.repeat(existing_reposizes, offsets, axis=-1)
                ts_indexes = np.repeat(existing_ts, offsets)
                ts_indexes = \
                    ts_len - 1 - np.flip(ts_indexes.reshape(len(counts), ts_len), axis=1).ravel()
                sums[missing] = sums[reposize_indexes, ts_indexes].reshape(sums.shape)[missing]
                counts[missing] = counts[reposize_indexes, ts_indexes].reshape(sums.shape)[missing]
            # average the individual backfilled means
            means = sums / counts
            if len(means):
                ts_means = np.mean(means, axis=0)
            else:
                ts_means = np.full(len(min_times), None, dtype=elapsed.dtype)
            # confidence intervals are not implemented
            metrics.append([Metric(m is not None, m, None, None) for m in ts_means.tolist()])

    def _values(self) -> List[List[Metric[timedelta]]]:
        return self._metrics

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        raise AssertionError("this must be never called")

    def _value(self, samples: np.ndarray) -> Metric[timedelta]:
        raise AssertionError("this must be never called")


@register_metric(CodeCheckMetricID.SUITES_PER_PR)
class SuitesPerPRCounter(AverageMetricCalculator[float]):
    """Average number of executed check suites per pull request metric."""

    may_have_negative_values = False
    dtype = float
    has_nan = True

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
        result = np.full((len(min_times), len(facts)), np.nan, dtype=np.float32)
        result[:, first_pr_encounters] = pr_suite_counts
        result[~mask_pr_times] = None
        return result


@register_metric(CodeCheckMetricID.SUITE_TIME_PER_PR)
class SuiteTimePerPRCalculator(AverageMetricCalculator[timedelta]):
    """Average check suite execution in PRs time metric."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"
    has_nan = True
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


class ConcurrencyCalculator(MetricCalculator[float]):
    """Calculate the concurrency value for each check run."""

    dtype = float
    has_nan = True
    is_pure_dependency = True

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        repos = facts[CheckRun.repository_full_name.key].values.astype("S")
        names = facts[CheckRun.name.key].values.astype("S")
        crtypes = np.char.add(np.char.add(repos, b"|"), names)
        del repos, names
        started_ats = facts[CheckRun.started_at.key].values.astype(min_times.dtype, copy=False)
        completed_ats = facts[CheckRun.completed_at.key].values.astype(min_times.dtype, copy=False)
        have_completed = completed_ats == completed_ats
        crtypes = crtypes[have_completed]
        time_order_started_ats = started_ats[have_completed]
        time_order_completed_ats = completed_ats[have_completed]
        _, crtypes_counts = np.unique(crtypes, return_counts=True)
        crtypes_order = np.argsort(crtypes)
        crtype_order_started_ats = time_order_started_ats[crtypes_order].astype("datetime64[s]")
        crtype_order_completed_ats = time_order_completed_ats[crtypes_order].astype("datetime64[s]")  # noqa
        intersections = calculate_interval_intersections(
            crtype_order_started_ats.view(np.uint64),
            crtype_order_completed_ats.view(np.uint64),
            np.cumsum(crtypes_counts),
        )
        intersections = intersections[np.argsort(crtypes_order)]
        result = np.full((len(min_times), len(facts)), np.nan, dtype=float)
        result[:, have_completed] = intersections
        mask = (min_times[:, None] <= started_ats) & (started_ats < max_times[:, None])
        result[~mask] = None
        return result

    def _value(self, samples: np.ndarray) -> Metric[float]:
        raise AssertionError("disabled for pure dependencies")


@register_metric(CodeCheckMetricID.CONCURRENCY)
class AvgConcurrencyCalculator(AverageMetricCalculator[float]):
    """Calculate the average concurrency of the check runs."""

    may_have_negative_values = False
    dtype = float
    has_nan = True
    deps = (ConcurrencyCalculator,)

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        return self._calcs[0].peek


@register_metric(CodeCheckMetricID.CONCURRENCY_MAX)
class MaxConcurrencyCalculator(MaxMetricCalculator[int]):
    """Calculate the maximum concurrency of the check runs."""

    may_have_negative_values = False
    dtype = float
    has_nan = True
    deps = (ConcurrencyCalculator,)

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        return self._calcs[0].peek

    def _agg(self, samples: np.ndarray) -> int:
        return int(super()._agg(samples))
