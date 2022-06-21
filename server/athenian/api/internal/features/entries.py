import asyncio
from datetime import datetime, timezone
from functools import partial, reduce
from itertools import chain
import logging
import pickle
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import aiomcache
import numpy as np
import pandas as pd
import sentry_sdk

from athenian.api import metadata
from athenian.api.async_utils import COROUTINE_YIELD_EVERY_ITER, gather
from athenian.api.cache import CancelCache, cached, cached_methods, short_term_exptime
from athenian.api.db import Database, add_pdb_hits, add_pdb_misses
from athenian.api.defer import defer
from athenian.api.internal.datetime_utils import coarsen_time_interval
from athenian.api.internal.features.code import CodeStats
from athenian.api.internal.features.github.check_run_metrics import (
    CheckRunBinnedHistogramCalculator,
    CheckRunBinnedMetricCalculator,
    group_check_runs_by_lines,
    group_check_runs_by_pushers,
    make_check_runs_count_grouper,
)
from athenian.api.internal.features.github.code import calc_code_stats
from athenian.api.internal.features.github.deployment_metrics import (
    DeploymentBinnedMetricCalculator,
    group_deployments_by_environments,
    group_deployments_by_participants,
    group_deployments_by_repositories,
)
from athenian.api.internal.features.github.developer_metrics import (
    DeveloperBinnedMetricCalculator,
    group_actions_by_developers,
)
from athenian.api.internal.features.github.pull_request_metrics import (
    PullRequestBinnedHistogramCalculator,
    PullRequestBinnedMetricCalculator,
    calculate_logical_prs_duplication_mask,
    group_prs_by_lines,
    group_prs_by_participants,
    need_jira_mapping,
)
from athenian.api.internal.features.github.release_metrics import (
    ReleaseBinnedMetricCalculator,
    calculate_logical_release_duplication_mask,
    group_releases_by_participants,
    merge_release_participants,
)
from athenian.api.internal.features.github.unfresh_pull_request_metrics import (
    UnfreshPullRequestFactsFetcher,
)
from athenian.api.internal.features.histogram import HistogramParameters
from athenian.api.internal.features.jira.issue_metrics import (
    IssuesLabelSplitter,
    JIRABinnedHistogramCalculator,
    JIRABinnedMetricCalculator,
    split_issues_by_participants,
)
from athenian.api.internal.features.metric_calculator import (
    DEFAULT_QUANTILE_STRIDE,
    MetricCalculatorEnsemble,
    group_by_repo,
    group_to_indexes,
)
from athenian.api.internal.jira import JIRAConfig
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.check_run import mine_check_runs
from athenian.api.internal.miners.github.commit import FilterCommitsProperty, extract_commits
from athenian.api.internal.miners.github.deployment import (
    load_jira_issues_for_deployments,
    mine_deployments,
)
from athenian.api.internal.miners.github.developer import (
    DeveloperTopic,
    developer_repository_column,
    mine_developer_activities,
)
from athenian.api.internal.miners.github.precomputed_prs import (
    DonePRFactsLoader,
    remove_ambiguous_prs,
    store_merged_unreleased_pull_request_facts,
    store_open_pull_request_facts,
    store_precomputed_done_facts,
)
from athenian.api.internal.miners.github.pull_request import (
    ImpossiblePullRequest,
    PullRequestFactsMiner,
    PullRequestMiner,
)
from athenian.api.internal.miners.github.release_mine import mine_releases
from athenian.api.internal.miners.jira.issue import (
    PullRequestJiraMapper,
    fetch_jira_issues,
    participant_columns,
)
from athenian.api.internal.miners.types import (
    JIRAParticipants,
    JIRAParticipationKind,
    PRParticipants,
    PRParticipationKind,
    PullRequestFacts,
    ReleaseFacts,
    ReleaseParticipants,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseMatch, ReleaseSettings
from athenian.api.models.metadata.github import CheckRun, PullRequest, PushCommit, Release
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import df_from_structs

unfresh_prs_threshold = 1000
unfresh_participants_threshold = 50


def _postprocess_cached_facts(
    result: Tuple[Dict[str, List[PullRequestFacts]], bool],
    with_jira_map: bool,
    **_,
) -> Tuple[Dict[str, List[PullRequestFacts]], bool]:
    if with_jira_map and not result[1]:
        raise CancelCache()
    return result


class UnsupportedMetricError(Exception):
    """Raised on attempt to calculate a histogram on a metric that's not possible."""


@cached_methods
class MetricEntriesCalculator:
    """Calculator for different metrics."""

    pr_miner = PullRequestMiner
    branch_miner = BranchMiner
    unfresh_pr_facts_fetcher = UnfreshPullRequestFactsFetcher
    pr_jira_mapper = PullRequestJiraMapper
    done_prs_facts_loader = DonePRFactsLoader
    load_delta = 0
    _log = logging.getLogger(f"{metadata.__package__}.MetricEntriesCalculator")

    def __init__(
        self,
        account: int,
        meta_ids: Tuple[int, ...],
        quantile_stride: int,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        cache: Optional[aiomcache.Client],
    ):
        """
        Create a `MetricEntriesCalculator`.

        :param quantile_stride: The cell size of the quantile grid in days.
        """
        self._account = account
        self._meta_ids = meta_ids
        self._quantile_stride = quantile_stride
        self._mdb = mdb
        self._pdb = pdb
        self._rdb = rdb
        self._cache = cache

    def is_ready_for(self, account: int, meta_ids: Tuple[int, ...]) -> bool:
        """Check whether the calculator is ready for the given account and meta ids."""
        return True

    @staticmethod
    def align_time_min_max(time_intervals, stride: int) -> Tuple[datetime, datetime]:
        """Widen the min and max timestamp so that it spans an integer number of several days."""
        ts_arr = np.array(
            [
                ts.replace(tzinfo=None)
                for ts in (
                    time_intervals
                    if isinstance(time_intervals[0], datetime)
                    else chain.from_iterable(time_intervals)
                )
            ],
            dtype="datetime64[s]",
        )
        if stride > 4 * 7:
            return (
                ts_arr.min().item().replace(tzinfo=timezone.utc),
                ts_arr.max().item().replace(tzinfo=timezone.utc),
            )
        aligned_ts_min, aligned_ts_max = MetricCalculatorEnsemble.compose_quantile_time_intervals(
            ts_arr.min(), ts_arr.max(), stride,
        )
        return (
            aligned_ts_min[0].item().replace(tzinfo=timezone.utc),
            aligned_ts_max[-1].item().replace(tzinfo=timezone.utc),
        )

    def _align_time_min_max(
        self,
        time_intervals,
        quantiles: Sequence[float],
    ) -> Tuple[datetime, datetime]:
        if quantiles[0] == 0 and quantiles[1] == 1:
            return self.align_time_min_max(time_intervals, 100500)
        return self.align_time_min_max(time_intervals, self._quantile_stride)

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda metrics, time_intervals, quantiles, lines, repositories, participants, labels, jira, environments, exclude_inactive, release_settings, logical_settings, **_: (  # noqa
            # TODO: use sorted(metrics) in cache key and then use a postprocess step to reorder
            # the cached value according to caller desired metrics order
            ",".join(metrics),
            ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
            ",".join(str(q) for q in quantiles),
            ",".join(str(n) for n in lines),
            _compose_cache_key_repositories(repositories),
            _compose_cache_key_participants(participants),
            labels,
            jira,
            ",".join(sorted(environments)),
            exclude_inactive,
            release_settings,
            logical_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def calc_pull_request_metrics_line_github(
        self,
        metrics: Sequence[str],
        time_intervals: Sequence[Sequence[datetime]],
        quantiles: Sequence[Union[float, int]],
        lines: Sequence[int],
        environments: Sequence[str],
        repositories: Sequence[Collection[str]],
        participants: List[PRParticipants],
        labels: LabelFilter,
        jira: JIRAFilter,
        exclude_inactive: bool,
        bots: Set[str],
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        branches: pd.DataFrame,
        default_branches: Dict[str, str],
        fresh: bool,
    ) -> np.ndarray:
        """
        Calculate pull request metrics on GitHub.

        :return: lines x repositories x participants x granularities x time intervals x metrics.
        """
        assert isinstance(repositories, (tuple, list))
        all_repositories, all_participants = _merge_repositories_and_participants(
            repositories, participants,
        )
        calc = PullRequestBinnedMetricCalculator(
            metrics,
            quantiles,
            self._quantile_stride,
            exclude_inactive=exclude_inactive,
            environments=environments,
        )
        time_from, time_to = self._align_time_min_max(time_intervals, quantiles)
        df_facts = await self.calc_pull_request_facts_github(
            time_from,
            time_to,
            all_repositories,
            all_participants,
            labels,
            jira,
            exclude_inactive,
            bots,
            release_settings,
            logical_settings,
            prefixer,
            fresh,
            need_jira_mapping(metrics),
            branches,
            default_branches,
        )
        lines_grouper = partial(group_prs_by_lines, lines)
        repo_grouper = partial(group_by_repo, PullRequest.repository_full_name.name, repositories)
        with_grouper = partial(group_prs_by_participants, participants)
        dedupe_mask = calculate_logical_prs_duplication_mask(
            df_facts, release_settings, logical_settings,
        )
        groups = group_to_indexes(
            df_facts,
            lines_grouper,
            repo_grouper,
            with_grouper,
            deduplicate_key=PullRequestFacts.f.node_id if dedupe_mask is not None else None,
            deduplicate_mask=dedupe_mask,
        )
        return calc(df_facts, time_intervals, groups)

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda defs, time_from, time_to, quantiles, lines, repositories, participants, labels, jira, environment, exclude_inactive, release_settings, logical_settings, **_: (  # noqa
            ",".join("%s:%s" % (k, sorted(v)) for k, v in sorted(defs.items())),
            time_from,
            time_to,
            ",".join(str(q) for q in quantiles),
            ",".join(str(n) for n in lines),
            _compose_cache_key_repositories(repositories),
            _compose_cache_key_participants(participants),
            labels,
            jira,
            environment,
            exclude_inactive,
            release_settings,
            logical_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def calc_pull_request_histograms_github(
        self,
        defs: Dict[HistogramParameters, List[str]],
        time_from: datetime,
        time_to: datetime,
        quantiles: Sequence[float],
        lines: Sequence[int],
        environment: Optional[str],
        repositories: Sequence[Collection[str]],
        participants: List[PRParticipants],
        labels: LabelFilter,
        jira: JIRAFilter,
        exclude_inactive: bool,
        bots: Set[str],
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        branches: Optional[pd.DataFrame],
        default_branches: Optional[Dict[str, str]],
        fresh: bool,
    ) -> np.ndarray:
        """
        Calculate the pull request histograms on GitHub.

        :return: defs x lines x repositories x participants -> List[Tuple[metric ID, Histogram]].
        """
        all_repositories, all_participants = _merge_repositories_and_participants(
            repositories, participants,
        )
        try:
            calc = PullRequestBinnedHistogramCalculator(
                defs.values(), quantiles, environments=[environment],
            )
        except KeyError as e:
            raise UnsupportedMetricError() from e
        df_facts = await self.calc_pull_request_facts_github(
            time_from,
            time_to,
            all_repositories,
            all_participants,
            labels,
            jira,
            exclude_inactive,
            bots,
            release_settings,
            logical_settings,
            prefixer,
            fresh,
            False,
            branches,
            default_branches,
        )
        lines_grouper = partial(group_prs_by_lines, lines)
        repo_grouper = partial(group_by_repo, PullRequest.repository_full_name.name, repositories)
        with_grouper = partial(group_prs_by_participants, participants)
        dedupe_mask = calculate_logical_prs_duplication_mask(
            df_facts, release_settings, logical_settings,
        )
        groups = group_to_indexes(
            df_facts,
            lines_grouper,
            repo_grouper,
            with_grouper,
            deduplicate_key=PullRequestFacts.f.node_id if dedupe_mask is not None else None,
            deduplicate_mask=dedupe_mask,
        )
        hists = calc(df_facts, [[time_from, time_to]], groups, defs)
        reshaped = np.full(hists.shape[:-1], None, object)
        reshaped_seq = reshaped.ravel()
        pos = 0
        for line_groups, metrics in zip(hists, defs.values()):
            for repo_groups in line_groups:
                for participants_groups in repo_groups:
                    for group_ts in participants_groups:
                        reshaped_seq[pos] = [(m, hist) for hist, m in zip(group_ts[0][0], metrics)]
                        pos += 1
        return reshaped

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda prop, time_intervals, repos, with_author, with_committer, only_default_branch, **kwargs: (  # noqa
            prop.value,
            ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
            ",".join(sorted(repos)),
            ",".join(sorted(with_author)) if with_author else "",
            ",".join(sorted(with_committer)) if with_committer else "",
            only_default_branch,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def calc_code_metrics_github(
        self,
        prop: FilterCommitsProperty,
        time_intervals: Sequence[datetime],
        repos: Collection[str],
        with_author: Optional[Collection[str]],
        with_committer: Optional[Collection[str]],
        only_default_branch: bool,
        prefixer: Prefixer,
    ) -> List[CodeStats]:
        """Filter code pushed on GitHub according to the specified criteria."""
        time_from, time_to = self.align_time_min_max(time_intervals, 100500)
        x_commits = await extract_commits(
            prop,
            time_from,
            time_to,
            repos,
            with_author,
            with_committer,
            only_default_branch,
            self.branch_miner(),
            prefixer,
            self._account,
            self._meta_ids,
            self._mdb,
            self._pdb,
            self._cache,
        )
        all_commits = await extract_commits(
            FilterCommitsProperty.NO_PR_MERGES,
            time_from,
            time_to,
            repos,
            with_author,
            with_committer,
            only_default_branch,
            self.branch_miner(),
            prefixer,
            self._account,
            self._meta_ids,
            self._mdb,
            self._pdb,
            self._cache,
            columns=[PushCommit.committed_date, PushCommit.additions, PushCommit.deletions],
        )
        return calc_code_stats(x_commits, all_commits, time_intervals)

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda devs, repositories, time_intervals, topics, labels, jira, release_settings, logical_settings, **_: (  # noqa
            ",".join(t.value for t in sorted(topics)),
            ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
            _compose_cache_key_repositories(repositories),
            _compose_cache_key_repositories(devs),  # yes, _repositories
            labels,
            jira,
            release_settings,
            logical_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def calc_developer_metrics_github(
        self,
        devs: Sequence[Collection[str]],
        repositories: Sequence[Collection[str]],
        time_intervals: Sequence[Sequence[datetime]],
        topics: Set[DeveloperTopic],
        labels: LabelFilter,
        jira: JIRAFilter,
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
    ) -> Tuple[np.ndarray, List[DeveloperTopic]]:
        """
        Calculate the developer metrics on GitHub.

        :return: repositories x granularities x devs x time intervals x topics.
        """
        all_devs = set(chain.from_iterable(devs))
        all_repos = set(chain.from_iterable(repositories))
        if not all_devs or not all_repos:
            return np.array([]), list(topics)
        time_from, time_to = self.align_time_min_max(time_intervals, 100500)
        mined_dfs = await mine_developer_activities(
            all_devs,
            all_repos,
            time_from,
            time_to,
            topics,
            labels,
            jira,
            release_settings,
            logical_settings,
            prefixer,
            self._account,
            self._meta_ids,
            self._mdb,
            self._pdb,
            self._rdb,
            self._cache,
        )
        topics_seq = []
        arrays = []
        repo_grouper = partial(group_by_repo, developer_repository_column, repositories)
        developer_grouper = partial(group_actions_by_developers, devs)
        for mined_topics, mined_df in mined_dfs:
            topics_seq.extend(mined_topics)
            calc = DeveloperBinnedMetricCalculator([t.value for t in mined_topics], (0, 1), 0)
            groups = group_to_indexes(mined_df, repo_grouper, developer_grouper)
            arrays.append(calc(mined_df, time_intervals, groups))
        result = np.full(arrays[0].shape, None)
        result.ravel()[:] = [
            [list(chain.from_iterable(m)) for m in zip(*lists)]
            for lists in zip(*(arr.ravel() for arr in arrays))
        ]
        result = result.swapaxes(1, 2)
        return result, topics_seq

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda metrics, time_intervals, quantiles, repositories, participants, labels, jira, release_settings, logical_settings, **_: (  # noqa
            ",".join(sorted(metrics)),
            ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
            ",".join(str(q) for q in quantiles),
            ",".join(str(sorted(r)) for r in repositories),
            ";".join(
                ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(p.items()))
                for p in participants
            ),
            labels,
            jira,
            release_settings,
            logical_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def calc_release_metrics_line_github(
        self,
        metrics: Sequence[str],
        time_intervals: Sequence[Sequence[datetime]],
        quantiles: Sequence[float],
        repositories: Sequence[Collection[str]],
        participants: List[ReleaseParticipants],
        labels: LabelFilter,
        jira: JIRAFilter,
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        branches: pd.DataFrame,
        default_branches: Dict[str, str],
    ) -> Tuple[np.ndarray, Dict[str, ReleaseMatch]]:
        """
        Calculate the release metrics on GitHub.

        :return: 1. participants x repositories x granularities x time intervals x metrics.
                 2. matched_bys - map from repository names to applied release matches.
        """
        time_from, time_to = self._align_time_min_max(time_intervals, quantiles)
        all_repositories = set(chain.from_iterable(repositories))
        calc = ReleaseBinnedMetricCalculator(metrics, quantiles, self._quantile_stride)
        all_participants = merge_release_participants(participants)
        releases, _, matched_bys, _ = await mine_releases(
            all_repositories,
            all_participants,
            branches,
            default_branches,
            time_from,
            time_to,
            labels,
            jira,
            release_settings,
            logical_settings,
            prefixer,
            self._account,
            self._meta_ids,
            self._mdb,
            self._pdb,
            self._rdb,
            self._cache,
            with_avatars=False,
            with_pr_titles=False,
        )
        df_facts = df_from_structs([f for _, f in releases])
        repo_grouper = partial(group_by_repo, Release.repository_full_name.name, repositories)
        participant_grouper = partial(group_releases_by_participants, participants)
        dedupe_mask = calculate_logical_release_duplication_mask(
            df_facts, release_settings, logical_settings,
        )
        groups = group_to_indexes(
            df_facts,
            participant_grouper,
            repo_grouper,
            deduplicate_key=ReleaseFacts.f.node_id if dedupe_mask is not None else None,
            deduplicate_mask=dedupe_mask,
        )
        values = calc(df_facts, time_intervals, groups)
        return values, matched_bys

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda metrics, time_intervals, quantiles, repositories, pushers, labels, jira, lines, logical_settings, **_: (  # noqa
            ",".join(sorted(metrics)),
            ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
            ",".join(str(q) for q in quantiles),
            ",".join(str(sorted(r)) for r in repositories),
            ";".join(",".join(g) for g in pushers),
            labels,
            jira,
            ",".join(str(i) for i in lines),
            logical_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def calc_check_run_metrics_line_github(
        self,
        metrics: Sequence[str],
        time_intervals: Sequence[Sequence[datetime]],
        quantiles: Sequence[float],
        repositories: Sequence[Collection[str]],
        pushers: List[Sequence[str]],
        split_by_check_runs: bool,
        labels: LabelFilter,
        jira: JIRAFilter,
        lines: Sequence[int],
        logical_settings: LogicalRepositorySettings,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the check run metrics on GitHub.

        :return: 1. pushers x repositories x lines x check runs count groups x granularities x time intervals x metrics.
                 2. how many suites in each check runs count group (meaningful only if split_by_check_runs=True).
                 3. suite sizes (meaningful only if split_by_check_runs=True).
        """  # noqa
        calc = CheckRunBinnedMetricCalculator(metrics, quantiles, self._quantile_stride)
        time_from, time_to = self._align_time_min_max(time_intervals, quantiles)
        (
            df_check_runs,
            groups,
            group_suite_counts,
            suite_sizes,
        ) = await self._mine_and_group_check_runs(
            time_from,
            time_to,
            repositories,
            pushers,
            split_by_check_runs,
            labels,
            jira,
            lines,
            logical_settings,
        )
        values = calc(df_check_runs, time_intervals, groups)
        return values, group_suite_counts, suite_sizes

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda defs, time_from, time_to, quantiles, repositories, pushers, labels, jira, lines, split_by_check_runs, logical_settings, **_: (  # noqa
            ",".join("%s:%s" % (k, sorted(v)) for k, v in sorted(defs.items())),
            time_from.timestamp(),
            time_to.timestamp(),
            ",".join(str(q) for q in quantiles),
            ",".join(str(sorted(r)) for r in repositories),
            ";".join(",".join(g) for g in pushers),
            labels,
            jira,
            ",".join(str(i) for i in lines),
            split_by_check_runs,
            logical_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def calc_check_run_histograms_line_github(
        self,
        defs: Dict[HistogramParameters, List[str]],
        time_from: datetime,
        time_to: datetime,
        quantiles: Sequence[float],
        repositories: Sequence[Collection[str]],
        pushers: List[Sequence[str]],
        split_by_check_runs: bool,
        labels: LabelFilter,
        jira: JIRAFilter,
        lines: Sequence[int],
        logical_settings: LogicalRepositorySettings,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate histograms over check runs on GitHub.

        :return: 1. defs x pushers x repositories x lines x check runs count groups -> List[Tuple[metric ID, Histogram]].
                 2. how many suites in each check runs count group (meaningful only if split_by_check_runs=True).
                 3. suite sizes (meaningful only if split_by_check_runs=True).
        """  # noqa
        try:
            calc = CheckRunBinnedHistogramCalculator(defs.values(), quantiles)
        except KeyError as e:
            raise UnsupportedMetricError() from e
        (
            df_check_runs,
            groups,
            group_suite_counts,
            suite_sizes,
        ) = await self._mine_and_group_check_runs(
            time_from,
            time_to,
            repositories,
            pushers,
            split_by_check_runs,
            labels,
            jira,
            lines,
            logical_settings,
        )
        hists = calc(df_check_runs, [[time_from, time_to]], groups, defs)
        reshaped = np.full(hists.shape[:-1], None, object)
        reshaped_seq = reshaped.ravel()
        pos = 0
        for commit_author_group, metrics in zip(hists, defs.values()):
            for repo_group in commit_author_group:
                for lines_group in repo_group:
                    for check_runs_group in lines_group:
                        for ts_group in check_runs_group:
                            reshaped_seq[pos] = [
                                (m, hist) for hist, m in zip(ts_group[0][0], metrics)
                            ]
                            pos += 1
        return reshaped, group_suite_counts, suite_sizes

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda metrics, time_intervals, quantiles, repositories, participants, environments, pr_labels, with_labels, without_labels, jira, release_settings, **_: (  # noqa
            ",".join(sorted(metrics)),
            ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
            ",".join(str(q) for q in quantiles),
            ",".join(str(sorted(r)) for r in repositories),
            ";".join(
                ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(p.items()))
                for p in participants
            ),
            ";".join(",".join(sorted(e)) for e in environments),
            pr_labels,
            with_labels,
            without_labels,
            jira,
            release_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def calc_deployment_metrics_line_github(
        self,
        metrics: Sequence[str],
        time_intervals: Sequence[Sequence[datetime]],
        quantiles: Sequence[float],
        repositories: Sequence[Collection[str]],
        participants: List[ReleaseParticipants],
        environments: List[List[str]],
        pr_labels: LabelFilter,
        with_labels: Mapping[str, Any],
        without_labels: Mapping[str, Any],
        jira: JIRAFilter,
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        branches: pd.DataFrame,
        default_branches: Dict[str, str],
        jira_ids: Optional[JIRAConfig],
    ) -> np.ndarray:
        """
        Calculate the deployment metrics on GitHub.

        :return: participants x repositories x environments x granularities x time intervals x metrics.
        """  # noqa
        time_from, time_to = self._align_time_min_max(time_intervals, quantiles)
        deps, _ = await mine_deployments(
            set(chain.from_iterable(repositories)),
            participants[0] if len(participants) == 1 else {},
            time_from,
            time_to,
            set(chain.from_iterable(environments)),
            [],
            with_labels,
            without_labels,
            pr_labels,
            jira,
            release_settings,
            logical_settings,
            branches,
            default_branches,
            prefixer,
            self._account,
            self._meta_ids,
            self._mdb,
            self._pdb,
            self._rdb,
            self._cache,
        )
        issues = await load_jira_issues_for_deployments(deps, jira_ids, self._meta_ids, self._mdb)
        calc = DeploymentBinnedMetricCalculator(
            metrics, quantiles, self._quantile_stride, jira=issues,
        )
        repo_grouper = partial(group_deployments_by_repositories, repositories)
        participant_grouper = partial(group_deployments_by_participants, participants)
        env_grouper = partial(group_deployments_by_environments, environments)
        groups = group_to_indexes(deps, participant_grouper, repo_grouper, env_grouper)
        values = calc(deps, time_intervals, groups)
        return values

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda metrics, time_intervals, quantiles, participants, label_filter, split_by_label, priorities, types, epics, exclude_inactive, release_settings, logical_settings, default_branches, **_: (  # noqa
            ",".join(sorted(metrics)),
            ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
            ",".join(str(q) for q in quantiles),
            ";".join(
                ",".join(f"{k.name}:{sorted(map(str, v))}" for k, v in sorted(p.items()))
                for p in participants
            ),
            label_filter,
            split_by_label,
            ",".join(sorted(priorities)),
            ",".join(sorted(types)),
            ",".join(sorted(epics) if not isinstance(epics, bool) else ["<flying>"]),
            exclude_inactive,
            release_settings,
            logical_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def calc_jira_metrics_line_github(
        self,
        metrics: Sequence[str],
        time_intervals: Sequence[Sequence[datetime]],
        quantiles: Sequence[float],
        participants: List[JIRAParticipants],
        label_filter: LabelFilter,
        split_by_label: bool,
        priorities: Collection[str],
        types: Collection[str],
        epics: Union[Collection[str], bool],
        exclude_inactive: bool,
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        default_branches: Dict[str, str],
        jira_ids: Optional[JIRAConfig],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the JIRA issue metrics.

        :return: 1. participants x labels x granularities x time intervals x metrics. \
                 2. Labels by which we grouped the issues.
        """
        time_from, time_to = self._align_time_min_max(time_intervals, quantiles)
        reporters, assignees, commenters = self._compile_jira_participants(participants)
        issues = await fetch_jira_issues(
            jira_ids,
            time_from,
            time_to,
            exclude_inactive,
            label_filter,
            priorities,
            types,
            epics,
            reporters,
            assignees,
            commenters,
            False,
            default_branches,
            release_settings,
            logical_settings,
            self._account,
            self._meta_ids,
            self._mdb,
            self._pdb,
            self._cache,
            extra_columns=participant_columns if len(participants) > 1 else (),
        )
        calc = JIRABinnedMetricCalculator(metrics, quantiles, self._quantile_stride)
        label_splitter = IssuesLabelSplitter(split_by_label, label_filter)
        groupers = partial(split_issues_by_participants, participants), label_splitter
        groups = group_to_indexes(issues, *groupers)
        return calc(issues, time_intervals, groups), label_splitter.labels

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda defs, time_from, time_to, quantiles, participants, label_filter, priorities, types, epics, exclude_inactive, release_settings, logical_settings, default_branches, **_: (  # noqa
            ",".join("%s:%s" % (k, sorted(v)) for k, v in sorted(defs.items())),
            time_from.timestamp(),
            time_to.timestamp(),
            ",".join(str(q) for q in quantiles),
            ";".join(
                ",".join(f"{k.name}:{sorted(map(str, v))}" for k, v in sorted(p.items()))
                for p in participants
            ),
            label_filter,
            ",".join(sorted(priorities)),
            ",".join(sorted(types)),
            ",".join(sorted(epics) if not isinstance(epics, bool) else ["<flying>"]),
            exclude_inactive,
            release_settings,
            logical_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def calc_jira_histograms(
        self,
        defs: Dict[HistogramParameters, List[str]],
        time_from: datetime,
        time_to: datetime,
        quantiles: Sequence[float],
        participants: List[JIRAParticipants],
        label_filter: LabelFilter,
        priorities: Collection[str],
        types: Collection[str],
        epics: Union[Collection[str], bool],
        exclude_inactive: bool,
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        default_branches: Dict[str, str],
        jira_ids: Optional[JIRAConfig],
    ) -> np.ndarray:
        """Calculate histograms over JIRA issues."""
        reporters, assignees, commenters = self._compile_jira_participants(participants)
        issues = await fetch_jira_issues(
            jira_ids,
            time_from,
            time_to,
            exclude_inactive,
            label_filter,
            priorities,
            types,
            epics,
            reporters,
            assignees,
            commenters,
            False,
            default_branches,
            release_settings,
            logical_settings,
            self._account,
            self._meta_ids,
            self._mdb,
            self._pdb,
            self._cache,
            extra_columns=participant_columns if len(participants) > 1 else (),
        )
        try:
            calc = JIRABinnedHistogramCalculator(defs.values(), quantiles)
        except KeyError as e:
            raise UnsupportedMetricError() from e
        with_groups = group_to_indexes(issues, partial(split_issues_by_participants, participants))
        return calc(issues, [[time_from, time_to]], with_groups, defs)

    @staticmethod
    def _compile_jira_participants(
        participants: List[JIRAParticipants],
    ) -> Tuple[Collection[str], Collection[str], Collection[str]]:
        reporters = list(
            set(
                chain.from_iterable(
                    ([p.lower() for p in g.get(JIRAParticipationKind.REPORTER, [])])
                    for g in participants
                ),
            ),
        )
        assignees = list(
            set(
                chain.from_iterable(
                    (
                        [
                            (p.lower() if p is not None else None)
                            for p in g.get(JIRAParticipationKind.ASSIGNEE, [])
                        ]
                    )
                    for g in participants
                ),
            ),
        )
        commenters = list(
            set(
                chain.from_iterable(
                    ([p.lower() for p in g.get(JIRAParticipationKind.COMMENTER, [])])
                    for g in participants
                ),
            ),
        )
        return reporters, assignees, commenters

    async def _mine_and_group_check_runs(
        self,
        time_from: datetime,
        time_to: datetime,
        repositories: Sequence[Collection[str]],
        pushers: List[Sequence[str]],
        split_by_check_runs: bool,
        labels: LabelFilter,
        jira: JIRAFilter,
        lines: Sequence[int],
        logical_settings: LogicalRepositorySettings,
    ) -> Tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        np.ndarray,  # groups  # how many suites in each group
    ]:  # distinct suite sizes
        all_repositories = set(chain.from_iterable(repositories))
        all_pushers = set(chain.from_iterable(pushers))
        df_check_runs = await mine_check_runs(
            time_from,
            time_to,
            all_repositories,
            all_pushers,
            labels,
            jira,
            logical_settings,
            self._meta_ids,
            self._mdb,
            self._cache,
        )
        repo_grouper = partial(group_by_repo, CheckRun.repository_full_name.name, repositories)
        commit_author_grouper = partial(group_check_runs_by_pushers, pushers)
        if split_by_check_runs:
            check_runs_grouper, suite_node_ids, suite_sizes = make_check_runs_count_grouper(
                df_check_runs,
            )
        else:
            suite_sizes = np.array([])

            def check_runs_grouper(df: pd.DataFrame) -> List[np.ndarray]:
                return [np.arange(len(df))]

        lines_grouper = partial(group_check_runs_by_lines, lines)
        groups = group_to_indexes(
            df_check_runs,
            commit_author_grouper,
            repo_grouper,
            lines_grouper,
            check_runs_grouper,
            deduplicate_key=CheckRun.check_run_node_id.name
            if logical_settings.has_logical_prs()
            else None,
        )
        group_suite_counts = np.zeros_like(groups, dtype=int)
        flat_group_suite_counts = group_suite_counts.ravel()
        if split_by_check_runs:
            for i, group in enumerate(groups.ravel()):
                flat_group_suite_counts[i] = len(np.unique(suite_node_ids[group]))
        return df_check_runs, groups, group_suite_counts, suite_sizes

    async def calc_pull_request_facts_github(
        self,
        time_from: datetime,
        time_to: datetime,
        repositories: Set[str],
        participants: PRParticipants,
        labels: LabelFilter,
        jira: JIRAFilter,
        exclude_inactive: bool,
        bots: Set[str],
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        fresh: bool,
        with_jira_map: bool,
        branches: Optional[pd.DataFrame] = None,
        default_branches: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate facts about pull request on GitHub.

        :param meta_ids: Metadata (GitHub) account IDs (*not the state DB account*) that own the
                         repos.
        :param exclude_inactive: Do not load PRs without events between `time_from` and `time_to`.
        :param fresh: If the number of done PRs for the time period and filters exceeds \
                      `unfresh_mode_threshold`, force querying mdb instead of pdb only.
        :return: PullRequestFacts packed in a Pandas DataFrame.
        """
        df, *_ = await self._calc_pull_request_facts_github(
            time_from,
            time_to,
            repositories,
            participants,
            labels,
            jira,
            exclude_inactive,
            bots,
            release_settings,
            logical_settings,
            prefixer,
            fresh,
            with_jira_map,
            branches=branches,
            default_branches=default_branches,
        )
        if df.empty:
            df = pd.DataFrame(columns=PullRequestFacts.f)
        return df

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda time_from, time_to, repositories, participants, labels, jira, exclude_inactive, release_settings, logical_settings, fresh, with_jira_map, **_: (  # noqa
            time_from,
            time_to,
            ",".join(sorted(repositories)),
            ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
            labels,
            jira,
            with_jira_map,
            exclude_inactive,
            release_settings,
            logical_settings,
            fresh,
        ),
        postprocess=_postprocess_cached_facts,
        cache=lambda self, **_: self._cache,
    )
    async def _calc_pull_request_facts_github(
        self,
        time_from: datetime,
        time_to: datetime,
        repositories: Set[str],
        participants: PRParticipants,
        labels: LabelFilter,
        jira: JIRAFilter,
        exclude_inactive: bool,
        bots: Set[str],
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        fresh: bool,
        with_jira_map: bool,
        branches: Optional[pd.DataFrame],
        default_branches: Optional[Dict[str, str]],
    ) -> Tuple[pd.DataFrame, bool]:
        assert isinstance(repositories, set)
        if branches is None or default_branches is None:
            branches, default_branches = await self.branch_miner.extract_branches(
                repositories, prefixer, self._meta_ids, self._mdb, self._cache,
            )
        precomputed_tasks = [
            self.done_prs_facts_loader.load_precomputed_done_facts_filters(
                time_from,
                time_to,
                repositories,
                participants,
                labels,
                default_branches,
                exclude_inactive,
                release_settings,
                prefixer,
                self._account,
                self._pdb,
            ),
        ]
        if exclude_inactive:
            precomputed_tasks.append(
                self.done_prs_facts_loader.load_precomputed_done_candidates(
                    time_from,
                    time_to,
                    repositories,
                    default_branches,
                    release_settings,
                    self._account,
                    self._pdb,
                ),
            )
        # augment blacklist with deployed PRs
        (precomputed_facts, ambiguous), *blacklist = await gather(*precomputed_tasks)
        precomputed_node_ids = {node_id for node_id, _ in precomputed_facts}
        if blacklist:
            blacklist = (blacklist[0][0] | precomputed_node_ids, blacklist[0][1])
        else:
            blacklist = (precomputed_node_ids, ambiguous)
        add_pdb_hits(self._pdb, "load_precomputed_done_facts_filters", len(precomputed_facts))

        prpk = PRParticipationKind
        if (
            (
                len(precomputed_facts) > unfresh_prs_threshold
                or len(participants.get(prpk.AUTHOR, [])) > unfresh_participants_threshold
            )
            and not fresh
            and not (participants.keys() - {prpk.AUTHOR, prpk.MERGER})
        ):
            facts = await self.unfresh_pr_facts_fetcher.fetch_pull_request_facts_unfresh(
                self.pr_miner,
                precomputed_facts,
                ambiguous,
                time_from,
                time_to,
                repositories,
                participants,
                labels,
                jira,
                self.pr_jira_mapper if with_jira_map else None,
                exclude_inactive,
                branches,
                default_branches,
                release_settings,
                logical_settings,
                prefixer,
                self._account,
                self._meta_ids,
                self._mdb,
                self._pdb,
                self._rdb,
                self._cache,
            )
            return df_from_structs(facts.values()), with_jira_map

        if with_jira_map:
            # schedule loading the PR->JIRA mapping
            done_jira_map_task = asyncio.create_task(
                self.pr_jira_mapper.load_pr_jira_mapping(
                    precomputed_node_ids, self._meta_ids, self._mdb,
                ),
                name="load_pr_jira_mapping/done",
            )
        done_deployments_task = asyncio.create_task(
            self.pr_miner.fetch_pr_deployments(
                precomputed_node_ids, self._account, self._pdb, self._rdb,
            ),
            name="fetch_pr_deployments/done",
        )
        add_pdb_misses(self._pdb, "fresh", 1)
        date_from, date_to = coarsen_time_interval(time_from, time_to)
        # the adjacent out-of-range pieces [date_from, time_from] and [time_to, date_to]
        # are effectively discarded later in BinnedMetricCalculator
        tasks = [
            self.pr_miner.mine(
                date_from,
                date_to,
                time_from,
                time_to,
                repositories,
                participants,
                labels,
                jira,
                with_jira_map,
                branches,
                default_branches,
                exclude_inactive,
                release_settings,
                logical_settings,
                prefixer,
                self._account,
                self._meta_ids,
                self._mdb,
                self._pdb,
                self._rdb,
                self._cache,
                pr_blacklist=blacklist,
            ),
        ]
        if jira and precomputed_facts:
            tasks.append(
                self.pr_miner.filter_jira(
                    precomputed_node_ids,
                    jira,
                    self._meta_ids,
                    self._mdb,
                    self._cache,
                    columns=[PullRequest.node_id],
                ),
            )
            (miner, unreleased_facts, matched_bys, unreleased_prs_event), filtered = await gather(
                *tasks, op="PullRequestMiner",
            )
            filtered_node_ids = set(filtered.index.values)
            precomputed_facts = {
                key: val for key, val in precomputed_facts.items() if key[0] in filtered_node_ids
            }
        else:
            miner, unreleased_facts, matched_bys, unreleased_prs_event = await tasks[0]
        new_deps = miner.dfs.deployments.copy()
        new_jira = miner.dfs.jiras.index.copy()
        precomputed_unreleased_prs = miner.drop(unreleased_facts)
        remove_ambiguous_prs(precomputed_facts, ambiguous, matched_bys)
        add_pdb_hits(self._pdb, "precomputed_unreleased_facts", len(precomputed_unreleased_prs))
        for key in precomputed_unreleased_prs:
            precomputed_facts[key] = unreleased_facts[key]
        facts_miner = PullRequestFactsMiner(bots)
        mined_prs = []
        mined_facts = {}
        open_pr_facts = []
        merged_unreleased_pr_facts = []
        done_count = 0
        with sentry_sdk.start_span(
            op="PullRequestMiner.__iter__ + PullRequestFactsMiner.__call__",
            description=str(len(miner)),
        ):
            for i, pr in enumerate(miner):
                if (i + 1) % COROUTINE_YIELD_EVERY_ITER == 0:
                    await asyncio.sleep(0)
                try:
                    facts = facts_miner(pr)
                except ImpossiblePullRequest:
                    continue
                mined_prs.append(pr)
                mined_facts[
                    (
                        pr.pr[PullRequest.node_id.name],
                        pr.pr[PullRequest.repository_full_name.name],
                    )
                ] = facts
                if facts.done:
                    done_count += 1
                elif not facts.closed:
                    open_pr_facts.append((pr, facts))
                else:
                    merged_unreleased_pr_facts.append((pr, facts))
        add_pdb_misses(self._pdb, "precomputed_done_facts", done_count)
        add_pdb_misses(self._pdb, "precomputed_open_facts", len(open_pr_facts))
        add_pdb_misses(
            self._pdb, "precomputed_merged_unreleased_facts", len(merged_unreleased_pr_facts),
        )
        add_pdb_misses(self._pdb, "facts", len(miner))
        if done_count > 0:
            # we don't care if exclude_inactive is True or False here
            # also, we dump all the `mined_facts`, the called function will figure out who's
            # released
            await defer(
                store_precomputed_done_facts(
                    mined_prs,
                    mined_facts.values(),
                    default_branches,
                    release_settings,
                    self._account,
                    self._pdb,
                ),
                "store_precomputed_done_facts(%d/%d)" % (done_count, len(miner)),
            )
        if len(open_pr_facts) > 0:
            await defer(
                store_open_pull_request_facts(open_pr_facts, self._account, self._pdb),
                "store_open_pull_request_facts(%d)" % len(open_pr_facts),
            )
        if len(merged_unreleased_pr_facts) > 0:
            await defer(
                store_merged_unreleased_pull_request_facts(
                    merged_unreleased_pr_facts,
                    matched_bys,
                    default_branches,
                    release_settings,
                    self._account,
                    self._pdb,
                    unreleased_prs_event,
                ),
                "store_merged_unreleased_pull_request_facts(%d)" % len(merged_unreleased_pr_facts),
            )
        tasks = [done_deployments_task]
        if with_jira_map:
            tasks.append(done_jira_map_task)
        done_deps, *done_jira_map = await gather(*tasks)
        if with_jira_map:
            jira_map = done_jira_map[0]
            for pr, jira_key in zip(
                new_jira.get_level_values(0).values, new_jira.get_level_values(1).values,
            ):
                jira_map.setdefault(pr, []).append(jira_key)
            for (node_id, _), f in precomputed_facts.items():
                try:
                    f.jira_ids = jira_map[node_id]
                except KeyError:
                    continue
        else:
            empty_list = []
            for f in precomputed_facts.values():
                f.jira_ids = empty_list
        self.unfresh_pr_facts_fetcher.append_deployments(
            precomputed_facts, pd.concat([done_deps, new_deps]), self._log,
        )
        all_facts_iter = chain(precomputed_facts.values(), mined_facts.values())
        all_facts_df = df_from_structs(
            all_facts_iter, length=len(precomputed_facts) + len(mined_facts),
        )
        return all_facts_df, with_jira_map


def _merge_repositories_and_participants(
    repositories: Sequence[Collection[str]],
    participants: List[PRParticipants],
) -> Tuple[Set[str], PRParticipants]:
    all_repositories = set(chain.from_iterable(repositories))
    if participants:
        all_participants = {}
        for k in PRParticipationKind:
            if kp := reduce(lambda x, y: x.union(y), [p.get(k, set()) for p in participants]):
                all_participants[k] = kp
    else:
        all_participants = {}
    return all_repositories, all_participants


def _compose_cache_key_repositories(repositories: Sequence[Collection[str]]) -> str:
    return ",".join(str(sorted(r)) for r in repositories)


def _compose_cache_key_participants(participants: List[PRParticipants]) -> str:
    return ";".join(
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(p.items())) for p in participants
    )


def make_calculator(
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> MetricEntriesCalculator:
    """Get the metrics calculator according to the account's features."""
    return MetricEntriesCalculator(
        account, meta_ids, DEFAULT_QUANTILE_STRIDE, mdb, pdb, rdb, cache,
    )
