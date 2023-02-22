from __future__ import annotations

import asyncio
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from itertools import chain
import logging
import os
import pickle
from typing import (
    Any,
    Callable,
    Collection,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

import aiomcache
import numpy as np
from numpy import typing as npt
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
    need_jira_mapping as pr_metrics_need_jira_mapping,
)
from athenian.api.internal.features.github.release_metrics import (
    ReleaseBinnedMetricCalculator,
    calculate_logical_release_duplication_mask,
    group_releases_by_participants,
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
    JIRAGrouping,
    MetricCalculatorEnsemble,
    deduplicate_groups,
    group_by_repo,
    group_jira_facts_by_jira,
    group_pr_facts_by_jira,
    group_release_facts_by_jira,
    group_to_indexes,
)
from athenian.api.internal.jira import JIRAConfig, JIRAEntitiesMapper
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
from athenian.api.internal.miners.participation import (
    JIRAParticipants,
    JIRAParticipationKind,
    PRParticipants,
    PRParticipationKind,
    ReleaseParticipants,
    ReleaseParticipationKind,
)
from athenian.api.internal.miners.types import JIRAEntityToFetch, PullRequestFacts, ReleaseFacts
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseMatch, ReleaseSettings
from athenian.api.models.metadata.github import (
    CheckRun,
    NodePullRequest,
    PullRequest,
    PushCommit,
    Release,
)
from athenian.api.models.metadata.jira import Issue
from athenian.api.models.persistentdata.models import HealthMetric
from athenian.api.pandas_io import deserialize_args, serialize_args
from athenian.api.sorted_ops import sorted_union1d
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import df_from_structs

unfresh_prs_threshold = 1000
unfresh_participants_threshold = 50
unfresh_branches_threshold = 1000


def _postprocess_cached_facts(
    result: tuple[dict[str, list[PullRequestFacts]], JIRAEntityToFetch | int, bool],
    with_jira: JIRAEntityToFetch | int,
    **_,
) -> tuple[dict[str, list[PullRequestFacts]], JIRAEntityToFetch | int, bool]:
    if (with_jira & result[1]) != with_jira:
        raise CancelCache()
    return (*result[:-1], False)


T = TypeVar("T")


class _CalcMetricsCache(Generic[T]):
    """Cache processing for calculator methods having a list of metrics as argument.

    Methods result must include a numpy array with elements ordered according to the order of
    metrics in arguments.
    Pre/post-processing functions in this class allows to share result cache across calls
    with different metrics order.

    """

    @classmethod
    def preprocess(cls, result: T, *, metrics: Sequence[str], **kwargs: Any) -> T:
        # cache value must be stored with sorted(metrics), adapt if needed
        if sorted(metrics) != metrics:
            sort_indexes = np.argsort(metrics)
            return cls.sort_metric_values_in_result(result, sort_indexes)
        return result

    @classmethod
    def postprocess(cls, result: T, *, metrics: Sequence[str], **kwargs: Any) -> T:
        # cache value has been stored with sorted(metrics), adapt before returning it to the caller
        if (sorted_metrics := sorted(metrics)) != metrics:
            sort_indexes = np.searchsorted(sorted_metrics, metrics)
            return cls.sort_metric_values_in_result(result, sort_indexes)
        return result

    @classmethod
    def sort_metric_values_in_result(cls, result: T, sort_indexes: Iterable[int]) -> T:
        raise NotImplementedError()

    @classmethod
    def _sorted_metric_values(cls, values: np.ndarray, sort_indexes: Iterable[int]) -> np.ndarray:
        flat_copy = values.ravel().copy()
        # each element of `flat_copy` is a list of list of metrics
        flat_copy[:] = [[[m[idx] for idx in sort_indexes] for m in obj] for obj in flat_copy]
        return flat_copy.reshape(values.shape)


class _PRMetricsLineGHCache(_CalcMetricsCache[np.ndarray]):
    @classmethod
    def sort_metric_values_in_result(
        cls,
        result: np.ndarray,
        sort_indexes: Iterable[int],
    ) -> np.ndarray:
        return cls._sorted_metric_values(result, sort_indexes)


class _ReleaseMetricsLineGHCache(_CalcMetricsCache[tuple[np.ndarray, dict[str, ReleaseMatch]]]):
    @classmethod
    def sort_metric_values_in_result(
        cls,
        result: tuple[np.ndarray, dict[str, ReleaseMatch]],
        sort_indexes: Iterable[int],
    ) -> tuple[np.ndarray, dict[str, ReleaseMatch]]:
        values = cls._sorted_metric_values(result[0], sort_indexes)
        return (values, result[1])


class _JIRAMetricsLineGHCache(_CalcMetricsCache[tuple[np.ndarray, np.ndarray]]):
    @classmethod
    def sort_metric_values_in_result(
        cls,
        result: tuple[np.ndarray, np.ndarray],
        sort_indexes: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        values = cls._sorted_metric_values(result[0], sort_indexes)
        return (values, result[1])


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
    _log = logging.getLogger(f"{metadata.__package__}.MetricEntriesCalculator")

    def __init__(
        self,
        account: int,
        meta_ids: tuple[int, ...],
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

    @staticmethod
    def align_time_min_max(time_intervals, stride: int) -> tuple[datetime, datetime]:
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
    ) -> tuple[datetime, datetime]:
        if quantiles[0] == 0 and quantiles[1] == 1:
            return self.align_time_min_max(time_intervals, 100500)
        return self.align_time_min_max(time_intervals, self._quantile_stride)

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda metrics, time_intervals, quantiles, lines, repositories, participants, labels, jiras, environments, exclude_inactive, release_settings, logical_settings, **_: (  # noqa
            # result can be cached independently from metrics order, see _PRMetricsLineGHCache
            ",".join(sorted(metrics)),
            ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
            ",".join(str(q) for q in quantiles),
            ",".join(str(n) for n in lines),
            _compose_cache_key_repositories(repositories),
            _compose_cache_key_participants(participants),
            labels,
            ",".join(str(j) for j in jiras),
            ",".join(sorted(environments)),
            exclude_inactive,
            release_settings,
            logical_settings,
        ),
        preprocess=_PRMetricsLineGHCache.preprocess,
        postprocess=_PRMetricsLineGHCache.postprocess,
        cache=lambda self, **_: self._cache,
    )
    async def calc_pull_request_metrics_line_github(
        self,
        metrics: Sequence[str],
        time_intervals: Sequence[Sequence[datetime]],
        quantiles: Sequence[float | int],
        lines: Sequence[int],
        environments: Sequence[str],
        repositories: Sequence[Collection[str]],
        participants: list[PRParticipants],
        labels: LabelFilter,
        jiras: Sequence[JIRAFilter],
        exclude_inactive: bool,
        bots: set[str],
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        branches: pd.DataFrame,
        default_branches: dict[str, str],
        fresh: bool,
    ) -> np.ndarray:
        """
        Calculate pull request metrics on GitHub.

        :return: jiras x lines x repositories x participants x
                 granularities x time intervals x metrics.
        """
        assert isinstance(repositories, (tuple, list))
        all_repositories = set(chain.from_iterable(repositories))
        all_participants = ParticipantsMerge.pr(participants)
        calc = PullRequestBinnedMetricCalculator(
            metrics,
            quantiles,
            self._quantile_stride,
            exclude_inactive=exclude_inactive,
            environments=environments,
        )
        time_from, time_to = self._align_time_min_max(time_intervals, quantiles)
        jiras = jiras or [JIRAFilter.empty()]
        jira_helper = await _PRsJIRAFiltersHelper.build(
            jiras, metrics, self._account, self._mdb, self._cache,
        )

        pr_facts_calc = self._build_pr_facts_calculator()
        df_facts = await pr_facts_calc(
            time_from,
            time_to,
            all_repositories,
            all_participants,
            labels,
            jira_helper.filter_,
            exclude_inactive,
            bots,
            release_settings,
            logical_settings,
            prefixer,
            fresh,
            jira_helper.entities_to_fetch,
            branches,
            default_branches,
        )

        lines_grouper = partial(group_prs_by_lines, lines)
        repo_grouper = partial(group_by_repo, PullRequest.repository_full_name.name, repositories)
        with_grouper = partial(group_prs_by_participants, participants, True)
        dedupe_mask = calculate_logical_prs_duplication_mask(
            df_facts, release_settings, logical_settings,
        )
        groups = group_to_indexes(
            df_facts,
            jira_helper.grouper,
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
        key=lambda requests, quantiles, exclude_inactive, bots, release_settings, logical_settings, **kwargs: (  # noqa
            ",".join(str(req) for req in requests),
            ",".join(str(q) for q in quantiles),
            exclude_inactive,
            release_settings,
            logical_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def batch_calc_pull_request_metrics_line_github(
        self,
        requests: Sequence[MetricsLineRequest],
        quantiles: Sequence[float | int],
        exclude_inactive: bool,
        bots: set[str],
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        branches: pd.DataFrame,
        default_branches: dict[str, str],
        fresh: bool,
    ) -> Sequence[np.ndarray]:
        """Execute a set of requests for pull request metrics.

        A numpy array is returned for every request in `requests`.
        The numpy array has these axes:
        - TeamSpecificFilters x granularities x intervals x metrics.

        Calling this method with multiple `requests` is more efficient than calling
        `calc_pull_request_metrics_line_github()` multiple times.
        """
        all_repositories = set(
            chain.from_iterable(t.repositories for request in requests for t in request.teams),
        )
        all_participants = ParticipantsMerge.pr(
            t.participants for request in requests for t in request.teams
        )
        assert all_repositories
        assert sum(bool(v) for v in all_participants.values())
        all_intervals = list(chain.from_iterable(request.time_intervals for request in requests))
        all_metrics = set(chain.from_iterable(request.metrics for request in requests))
        time_from, time_to = self._align_time_min_max(all_intervals, quantiles)

        jira_filters = list(chain.from_iterable(req.all_jira_filters() for req in requests))
        convert_jira_filters_to_grouping = await _JIRAFilterToGroupingConverter.build(
            jira_filters, self._account, self._mdb, self._cache,
        )
        jira_filter = JIRAFilter.combine(*jira_filters)

        jira_entities_to_fetch = _get_jira_entities_to_fetch(jira_filters, all_metrics)

        pr_facts_calc = self._build_pr_facts_calculator()
        df_facts = await pr_facts_calc(
            time_from,
            time_to,
            all_repositories,
            all_participants,
            LabelFilter.empty(),
            jira_filter,
            exclude_inactive,
            bots,
            release_settings,
            logical_settings,
            prefixer,
            fresh,
            jira_entities_to_fetch,
            branches,
            default_branches,
        )

        dedupe_mask = calculate_logical_prs_duplication_mask(
            df_facts, release_settings, logical_settings,
        )

        repo_grouping = _BatchRepoGrouping(all_repositories, df_facts, logical_settings)
        results = [None] * len(requests)
        executor = ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(requests)))

        def calc_request(i: int, request: MetricsLineRequest) -> None:
            jira_grouping = convert_jira_filters_to_grouping(t.jira_filter for t in request.teams)
            group_by = [
                repo_grouping.get_groups([t.repositories for t in request.teams]),
                group_prs_by_participants(
                    [t.participants for t in request.teams], False, df_facts,
                ),
                group_pr_facts_by_jira(jira_grouping, df_facts),
            ]
            groups = _intersect_items_groups(len(request.teams), len(df_facts), *group_by)
            groups = deduplicate_groups(
                groups,
                df_facts,
                deduplicate_key=PullRequestFacts.f.node_id if dedupe_mask is not None else None,
                deduplicate_mask=dedupe_mask,
            )
            calc = PullRequestBinnedMetricCalculator(
                request.metrics,
                quantiles,
                self._quantile_stride,
                exclude_inactive=exclude_inactive,
                **request.metric_params,
                # environments=request.environments,
            )
            results[i] = calc(df_facts, request.time_intervals, groups)

        with sentry_sdk.start_span(
            op="PullRequestBinnedMetricCalculator",
            description=str(len(requests)),
        ):
            for i, request in enumerate(requests):
                executor.submit(calc_request, i, request)

            executor.shutdown(wait=True)
        return results

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
        defs: dict[HistogramParameters, list[str]],
        time_from: datetime,
        time_to: datetime,
        quantiles: Sequence[float],
        lines: Sequence[int],
        environment: Optional[str],
        repositories: Sequence[Collection[str]],
        participants: list[PRParticipants],
        labels: LabelFilter,
        jira: JIRAFilter,
        exclude_inactive: bool,
        bots: set[str],
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        branches: Optional[pd.DataFrame],
        default_branches: Optional[dict[str, str]],
        fresh: bool,
    ) -> np.ndarray:
        """
        Calculate the pull request histograms on GitHub.

        :return: defs x lines x repositories x participants -> list[tuple[metric ID, Histogram]].
        """
        all_repositories = set(chain.from_iterable(repositories))
        all_participants = ParticipantsMerge.pr(participants)
        try:
            calc = PullRequestBinnedHistogramCalculator(
                defs.values(), quantiles, environments=[environment],
            )
        except KeyError as e:
            raise UnsupportedMetricError() from e
        pr_facts_calc = self._build_pr_facts_calculator()
        df_facts = await pr_facts_calc(
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
            JIRAEntityToFetch.NOTHING,
            branches,
            default_branches,
        )
        lines_grouper = partial(group_prs_by_lines, lines)
        repo_grouper = partial(group_by_repo, PullRequest.repository_full_name.name, repositories)
        with_grouper = partial(group_prs_by_participants, participants, True)
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
    ) -> list[CodeStats]:
        """Filter code pushed on GitHub according to the specified criteria."""
        time_from, time_to = self.align_time_min_max(time_intervals, 100500)
        x_commits, all_commits = await gather(
            extract_commits(
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
                None,
                self._cache,
                with_deployments=False,
            ),
            extract_commits(
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
                None,
                self._cache,
                columns=[PushCommit.committed_date, PushCommit.additions, PushCommit.deletions],
                with_deployments=False,
            ),
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
        topics: set[DeveloperTopic],
        labels: LabelFilter,
        jira: JIRAFilter,
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
    ) -> tuple[np.ndarray, list[DeveloperTopic]]:
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
        arrays = [None] * len(mined_dfs)
        repo_grouper = partial(group_by_repo, developer_repository_column, repositories)
        developer_grouper = partial(group_actions_by_developers, devs)
        executor = ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(mined_dfs)))

        def calculate(i: int) -> None:
            mined_topics, mined_df = mined_dfs[i]
            calc = DeveloperBinnedMetricCalculator([t.value for t in mined_topics], (0, 1), 0)
            groups = group_to_indexes(mined_df, repo_grouper, developer_grouper)
            arrays[i] = calc(mined_df, time_intervals, groups)

        with sentry_sdk.start_span(
            op="DeveloperBinnedMetricCalculator",
            description=str(len(mined_dfs)),
        ):
            for i, (mined_topics, _) in enumerate(mined_dfs):
                topics_seq.extend(mined_topics)
                executor.submit(calculate, i)

            executor.shutdown(wait=True)

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
        key=lambda metrics, time_intervals, quantiles, repositories, participants, labels, jiras, release_settings, logical_settings, **_: (  # noqa
            ",".join(sorted(metrics)),
            ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
            ",".join(str(q) for q in quantiles),
            ",".join(str(sorted(r)) for r in repositories),
            _compose_cache_key_participants(participants),
            labels,
            ",".join(str(j) for j in jiras),
            release_settings,
            logical_settings,
        ),
        preprocess=_ReleaseMetricsLineGHCache.preprocess,
        postprocess=_ReleaseMetricsLineGHCache.postprocess,
        cache=lambda self, **_: self._cache,
    )
    async def calc_release_metrics_line_github(
        self,
        metrics: Sequence[str],
        time_intervals: Sequence[Sequence[datetime]],
        quantiles: Sequence[float],
        repositories: Sequence[Collection[str]],
        participants: list[ReleaseParticipants],
        labels: LabelFilter,
        jiras: Sequence[JIRAFilter],
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        branches: pd.DataFrame,
        default_branches: dict[str, str],
    ) -> tuple[np.ndarray, dict[str, ReleaseMatch]]:
        """
        Calculate the release metrics on GitHub.

        :return: 1. jiras x participants x repositories x granularities x time intervals x metrics.
                 2. matched_bys - map from repository names to applied release matches.
        """
        time_from, time_to = self._align_time_min_max(time_intervals, quantiles)
        all_repositories = set(chain.from_iterable(repositories))
        calc = ReleaseBinnedMetricCalculator(metrics, quantiles, self._quantile_stride)
        all_participants = ParticipantsMerge.release(participants)
        jiras = jiras or [JIRAFilter.empty()]
        jira_helper = await _ReleaseJIRAFiltersHelper.build(
            jiras, self._account, self._mdb, self._cache,
        )

        df_facts, _, matched_bys, _ = await mine_releases(
            all_repositories,
            all_participants,
            branches,
            default_branches,
            time_from,
            time_to,
            labels,
            jira_helper.filter_,
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
            with_extended_pr_details=False,
            with_jira=jira_helper.entities_to_fetch,
        )

        repo_grouper = partial(group_by_repo, Release.repository_full_name.name, repositories)
        participant_grouper = partial(group_releases_by_participants, participants)
        dedupe_mask = calculate_logical_release_duplication_mask(
            df_facts, release_settings, logical_settings,
        )
        groups = group_to_indexes(
            df_facts,
            jira_helper.grouper,
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
        key=lambda requests, quantiles, release_settings, logical_settings, **kwargs: (
            ",".join(str(req) for req in requests),
            ",".join(str(q) for q in quantiles),
            release_settings,
            logical_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def batch_calc_release_metrics_line_github(
        self,
        requests: Sequence[MetricsLineRequest],
        quantiles: Sequence[float],
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        branches: pd.DataFrame,
        default_branches: dict[str, str],
    ) -> Sequence[np.ndarray]:
        """Execute a set of requests for release metrics.

        A numpy array is returned for every request in `requests`.
        The numpy array has these axes:
        - TeamSpecificFilters x granularities x intervals x metrics.

        Calling this method with multiple `requests` is more efficient than calling
        `calc_release_metrics_line_github()` multiple times.
        """
        all_intervals = list(chain.from_iterable(request.time_intervals for request in requests))
        time_from, time_to = self._align_time_min_max(all_intervals, quantiles)
        all_repositories = set(
            chain.from_iterable(t.repositories for request in requests for t in request.teams),
        )
        all_participants = ParticipantsMerge.release(
            t.participants for request in requests for t in request.teams
        )
        jira_filters = list(chain.from_iterable(req.all_jira_filters() for req in requests))
        convert_jira_filters_to_grouping = await _JIRAFilterToGroupingConverter.build(
            jira_filters, self._account, self._mdb, self._cache,
        )
        jira_filter = JIRAFilter.combine(*jira_filters)

        df_facts, _, _, _ = await mine_releases(
            all_repositories,
            all_participants,
            branches,
            default_branches,
            time_from,
            time_to,
            LabelFilter.empty(),
            jira_filter,
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
            with_extended_pr_details=False,
            with_jira=_get_jira_entities_to_fetch(jira_filters, ()),
        )

        repo_grouping = _BatchRepoGrouping(all_repositories, df_facts, logical_settings)
        results = [None] * len(requests)

        executor = ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(requests)))

        def calc_request(i: int, request: MetricsLineRequest) -> None:
            jira_grouping = convert_jira_filters_to_grouping(t.jira_filter for t in request.teams)
            group_by = [
                repo_grouping.get_groups([t.repositories for t in request.teams]),
                group_releases_by_participants([t.participants for t in request.teams], df_facts),
                group_release_facts_by_jira(jira_grouping, df_facts),
            ]
            groups = _intersect_items_groups(len(request.teams), len(df_facts), *group_by)

            dedupe_mask = calculate_logical_release_duplication_mask(
                df_facts, release_settings, logical_settings,
            )
            groups = deduplicate_groups(
                groups,
                df_facts,
                deduplicate_key=ReleaseFacts.f.node_id if dedupe_mask is not None else None,
                deduplicate_mask=dedupe_mask,
            )
            calc = ReleaseBinnedMetricCalculator(
                request.metrics, quantiles, self._quantile_stride, **request.metric_params,
            )
            results[i] = calc(df_facts, request.time_intervals, groups)

        with sentry_sdk.start_span(
            op="ReleaseBinnedMetricCalculator",
            description=str(len(requests)),
        ):
            for i, request in enumerate(requests):
                executor.submit(calc_request, i, request)

            executor.shutdown(wait=True)
        return results

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
        pushers: list[Sequence[str]],
        split_by_check_runs: bool,
        labels: LabelFilter,
        jira: JIRAFilter,
        lines: Sequence[int],
        logical_settings: LogicalRepositorySettings,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        defs: dict[HistogramParameters, list[str]],
        time_from: datetime,
        time_to: datetime,
        quantiles: Sequence[float],
        repositories: Sequence[Collection[str]],
        pushers: list[Sequence[str]],
        split_by_check_runs: bool,
        labels: LabelFilter,
        jira: JIRAFilter,
        lines: Sequence[int],
        logical_settings: LogicalRepositorySettings,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate histograms over check runs on GitHub.

        :return: 1. defs x pushers x repositories x lines x check runs count groups ->
                      list[tuple[metric ID, Histogram]].
                 2. how many suites in each check runs count group
                      (meaningful only if split_by_check_runs=True).
                 3. suite sizes (meaningful only if split_by_check_runs=True).
        """
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
        participants: list[ReleaseParticipants],
        environments: list[list[str]],
        pr_labels: LabelFilter,
        with_labels: Mapping[str, Any],
        without_labels: Mapping[str, Any],
        jira: JIRAFilter,
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        branches: pd.DataFrame,
        default_branches: dict[str, str],
        jira_ids: Optional[JIRAConfig],
    ) -> np.ndarray:
        """
        Calculate the deployment metrics on GitHub.

        :return: participants x repositories x environments x granularities x time intervals x metrics.
        """  # noqa
        time_from, time_to = self._align_time_min_max(time_intervals, quantiles)
        deps = await mine_deployments(
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
            jira_ids,
            self._meta_ids,
            self._mdb,
            self._pdb,
            self._rdb,
            self._cache,
            with_jira=True,
        )
        issues = await load_jira_issues_for_deployments(deps, jira_ids, self._mdb)
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
        key=lambda metrics, time_intervals, quantiles, participants, groups, split_by_label, exclude_inactive, release_settings, logical_settings, default_branches, **_: (  # noqa
            ",".join(sorted(metrics)),
            ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
            ",".join(str(q) for q in quantiles),
            # don't use _compose_cache_key_participants, it doesn't bear None-s
            ";".join(
                ",".join(f"{k.name}:{sorted(map(str, v))}" for k, v in sorted(p.items()))
                for p in participants
            ),
            ",".join(str(g) for g in groups),
            split_by_label,
            exclude_inactive,
            release_settings,
            logical_settings,
        ),
        preprocess=_JIRAMetricsLineGHCache.preprocess,
        postprocess=_JIRAMetricsLineGHCache.postprocess,
        cache=lambda self, **_: self._cache,
    )
    async def calc_jira_metrics_line_github(
        self,
        metrics: Sequence[str],
        time_intervals: Sequence[Sequence[datetime]],
        quantiles: Sequence[float],
        participants: list[JIRAParticipants],
        groups: Sequence[JIRAFilter],
        split_by_label: bool,
        exclude_inactive: bool,
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        default_branches: dict[str, str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the JIRA issue metrics.

        :return: 1. participants x groups x granularities x time intervals x metrics. \
                 2. Labels by which we grouped the issues.
        """
        time_from, time_to = self._align_time_min_max(time_intervals, quantiles)
        reporters, assignees, commenters = ParticipantsMerge.jira(participants)

        # there should be at least one group, and every should include a valid jira account id
        assert len(groups)
        assert len(all_accounts := {g.account for g in groups}) == 1 and all_accounts.pop()

        jira_entities = _get_jira_entities_to_fetch(groups, ())
        extra_columns = JIRAEntityToFetch.to_columns(jira_entities)
        if len(participants) > 1:
            extra_columns.extend(participant_columns)

        label_splitter = IssuesLabelSplitter(split_by_label, groups[0].labels)
        if split_by_label:
            assert len(groups) == 1
            if Issue.labels not in extra_columns:
                extra_columns.append(Issue.labels)
            jira_filter_grouper = label_splitter
        elif len(groups) == 1:
            jira_filter_grouper = _global_grouper
        else:
            filters_to_grouping = await _JIRAFilterToGroupingConverter.build(
                groups, self._account, self._mdb, self._cache,
            )
            jira_filter_grouper = partial(group_jira_facts_by_jira, filters_to_grouping(groups))

        issues = await fetch_jira_issues(
            time_from,
            time_to,
            JIRAFilter.combine(*groups),
            exclude_inactive,
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
            extra_columns=extra_columns,
        )
        calc = JIRABinnedMetricCalculator(metrics, quantiles, self._quantile_stride)

        groupers = partial(split_issues_by_participants, participants), jira_filter_grouper
        groups = group_to_indexes(issues, *groupers)
        return calc(issues, time_intervals, groups), label_splitter.labels

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda requests, quantiles, exclude_inactive, release_settings, logical_settings, **_: (  # noqa
            ",".join(str(req) for req in requests),
            ",".join(str(q) for q in quantiles),
            exclude_inactive,
            release_settings,
            logical_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def batch_calc_jira_metrics_line_github(
        self,
        requests: Sequence[MetricsLineRequest],
        quantiles: Sequence[float],
        exclude_inactive: bool,
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        default_branches: dict[str, str],
        jira_ids: JIRAConfig,
    ) -> Sequence[np.ndarray]:
        """
        Execute a set of requests for jira metrics.

        A numpy array is returned for every request in `requests`.
        The numpy array has these axes:
        - TeamSpecificFilters x granularities x intervals x metrics.

        Calling this method with multiple `requests` is more efficient than calling
        `calc_jira_metrics_line_github()` multiple times.
        """
        all_intervals = list(chain.from_iterable(request.time_intervals for request in requests))
        time_from, time_to = self._align_time_min_max(all_intervals, quantiles)
        extra_columns = list(participant_columns)
        reporters, assignees, commenters = ParticipantsMerge.jira(
            [t.participants for request in requests for t in request.teams],
        )

        jira_filters = list(chain.from_iterable(req.all_jira_filters() for req in requests))
        convert_jira_filters_to_grouping = await _JIRAFilterToGroupingConverter.build(
            jira_filters, self._account, self._mdb, self._cache,
        )
        if any(jira_filters):
            # if any filter is True-ish group_jira_facts_by_jira will need the extra info to group
            extra_columns.extend([Issue.type_id, Issue.priority_id, Issue.project_id])

        jira_filter = JIRAFilter.combine(*jira_filters) or JIRAFilter.from_jira_config(jira_ids)

        assert reporters or assignees or commenters
        issues = await fetch_jira_issues(
            time_from,
            time_to,
            jira_filter,
            exclude_inactive,
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
            extra_columns=extra_columns,
        )

        results = [None] * len(requests)
        executor = ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(requests)))

        def calc_request(i: int, request: MetricsLineRequest) -> None:
            jira_grouping = convert_jira_filters_to_grouping(t.jira_filter for t in request.teams)
            group_by = [
                split_issues_by_participants([t.participants for t in request.teams], issues),
                group_jira_facts_by_jira(jira_grouping, issues),
            ]
            groups = _intersect_items_groups(len(request.teams), len(issues), *group_by)
            calc = JIRABinnedMetricCalculator(
                request.metrics, quantiles, self._quantile_stride, **request.metric_params,
            )
            results[i] = calc(issues, request.time_intervals, groups)

        with sentry_sdk.start_span(
            op="JIRABinnedMetricCalculator",
            description=str(len(requests)),
        ):
            for i, request in enumerate(requests):
                executor.submit(calc_request, i, request)

            executor.shutdown(wait=True)
        return results

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda defs, time_from, time_to, quantiles, participants, label_filter, priorities, types, epics, exclude_inactive, release_settings, logical_settings, default_branches, jira_ids, **_: (  # noqa
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
            ",".join(sorted(jira_ids.projects)),
            ",".join(sorted(epics) if not isinstance(epics, bool) else ["<flying>"]),
            exclude_inactive,
            release_settings,
            logical_settings,
        ),
        cache=lambda self, **_: self._cache,
    )
    async def calc_jira_histograms(
        self,
        defs: dict[HistogramParameters, list[str]],
        time_from: datetime,
        time_to: datetime,
        quantiles: Sequence[float],
        participants: list[JIRAParticipants],
        label_filter: LabelFilter,
        priorities: Collection[str],
        types: Collection[str],
        epics: Collection[str] | bool,
        exclude_inactive: bool,
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        default_branches: dict[str, str],
        jira_ids: Optional[JIRAConfig],
    ) -> np.ndarray:
        """Calculate histograms over JIRA issues."""
        reporters, assignees, commenters = ParticipantsMerge.jira(participants)
        jira_filters = JIRAFilter.from_jira_config(jira_ids).replace(
            labels=label_filter, issue_types=types, epics=epics, priorities=priorities,
        )
        issues = await fetch_jira_issues(
            time_from,
            time_to,
            jira_filters,
            exclude_inactive,
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

    async def _mine_and_group_check_runs(
        self,
        time_from: datetime,
        time_to: datetime,
        repositories: Sequence[Collection[str]],
        pushers: list[Sequence[str]],
        split_by_check_runs: bool,
        labels: LabelFilter,
        jira: JIRAFilter,
        lines: Sequence[int],
        logical_settings: LogicalRepositorySettings,
    ) -> tuple[
        pd.DataFrame,
        np.ndarray,  # groups
        np.ndarray,  # how many suites in each group
        np.ndarray,  # distinct suite sizes
    ]:
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

            def check_runs_grouper(df: pd.DataFrame) -> list[np.ndarray]:
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

    def _build_pr_facts_calculator(self) -> PRFactsCalculator:
        return PRFactsCalculator(
            self._account,
            self._meta_ids,
            self._mdb,
            self._pdb,
            self._rdb,
            self.pr_miner,
            self.branch_miner,
            self.done_prs_facts_loader,
            self.unfresh_pr_facts_fetcher,
            self.pr_jira_mapper,
            self._cache,
        )


class _BatchRepoGrouping:
    """Handles the grouping by repositories for batch metrics calculation."""

    def __init__(
        self,
        all_repositories: set[str],
        df_facts: pd.Dataframe,
        logical_settings: LogicalRepositorySettings,
    ):
        self._all_repositories = all_repositories
        self._df_facts = df_facts
        self._augment_with_logical = logical_settings.augment_with_logical_repos

    def get_groups(self, request_repos: Collection[Collection[str]]) -> list[npt.NDarray[int]]:
        """Get the groups for the given repository sets."""
        # when every repo set is equal to `all_repositories` we can just select all `df_facts`
        # since `all_repositories` is the union of every repo set received here
        # it's faster to just compare the lengths
        if all(len(r) == len(self._all_repositories) for r in request_repos):
            return [np.arange(len(self._df_facts))] * len(request_repos)
        else:
            augmented_request_repos = [self._augment_with_logical(r) for r in request_repos]
            return group_by_repo(
                PullRequest.repository_full_name.name, augmented_request_repos, self._df_facts,
            )


class _JIRAFilterToGroupingConverter:
    """Converter of JIRAFilter objects to JIRAGrouping-s.

    Instances should be created with `build()`, which pre-scans all groups to be handled
    and avoids to build `entities_mapper` if possible.
    `jira_filters` passed to each __call__ should be part of `all_filters_ passed to `build()`.

    """

    def __init__(self, entities_mapper: Optional[JIRAEntitiesMapper]):
        self._entities_mapper = entities_mapper

    @classmethod
    async def build(
        cls,
        all_filters: Iterable[JIRAFilter],
        account: int,
        mdb: Database,
        cache: Optional[aiomcache.Client],
    ) -> _JIRAFilterToGroupingConverter:
        actual_filters = [f for f in all_filters if f]
        if any((f.priorities or f.issue_types) for f in actual_filters):
            jira_acc_ids = [f.account for f in actual_filters]
            if len(set(jira_acc_ids)) > 1:
                raise ValueError("JIRAFilters about multiple Jira installations not supported")
            entities_mapper = await JIRAEntitiesMapper.load(jira_acc_ids[0], mdb)
        else:
            entities_mapper = None
        return cls(entities_mapper)

    def __call__(self, jira_filters: Iterable[JIRAFilter]) -> list[JIRAGrouping]:
        groups = []
        for jira_filter in jira_filters:
            if jira_filter:
                projects = jira_filter.projects if jira_filter.custom_projects else None
                if jira_filter.priorities:
                    priorities = self._mapper.translate_priority_names(jira_filter.priorities)
                else:
                    priorities = None
                if jira_filter.issue_types:
                    issue_types = self._mapper.translate_types(jira_filter.issue_types)
                else:
                    issue_types = None
                # jira_filter.labels.exclude are ignored for grouping
                labels = jira_filter.labels.include or None
                group = JIRAGrouping(projects, priorities, issue_types, labels)
            else:  # this includes when custom_project is False and only projects is present
                group = JIRAGrouping.empty()
            groups.append(group)

        return groups

    @property
    def _mapper(self) -> JIRAEntitiesMapper:
        if self._entities_mapper is None:
            raise ValueError("A JIRAEntitiesMapper is required to convert a filled JIRAFilter")
        return self._entities_mapper


@dataclasses.dataclass
class _FactsJIRAFiltersHelper:
    """Collect the needed information to correctly mine and group facts by JIRAFilter-s."""

    filter_: JIRAFilter
    """The single filter to use to mine facts."""
    entities_to_fetch: JIRAEntityToFetch
    """The entities to fetch in order to correctly apply grouping on mined facts"""
    grouper: Callable[[pd.DataFrame], list[np.ndarray]]
    """The grouping function to group by jira filters."""


class _PRsJIRAFiltersHelper(_FactsJIRAFiltersHelper):
    """Collect the needed information to correctly fetch and group prs by JIRAFilter-s."""

    @classmethod
    async def build(
        cls,
        jiras: Sequence[JIRAFilter],
        metrics: Sequence[str],
        account: int,
        mdb: Database,
        cache: aiomcache.Client | None,
    ) -> _PRsJIRAFiltersHelper:
        filter_ = JIRAFilter.combine(*jiras)
        if len(jiras) == 1:
            # if there's a single JIRAFilter the global `filter_` used to mine/fetch
            # will only include releases matching the single filter, so we can avoid grouping
            # and fetch no jira entities
            with_jira = _get_jira_entities_to_fetch((), metrics)
            grouper = _global_grouper
        else:
            with_jira = _get_jira_entities_to_fetch(jiras, metrics)
            filters_to_grouping = await _JIRAFilterToGroupingConverter.build(
                jiras, account, mdb, cache,
            )
            grouper = partial(group_pr_facts_by_jira, filters_to_grouping(jiras))
        return cls(filter_, with_jira, grouper)


class _ReleaseJIRAFiltersHelper(_FactsJIRAFiltersHelper):
    """Collect the needed information to correctly fetch and group releases by JIRAFilter-s."""

    @classmethod
    async def build(
        cls,
        jiras: Sequence[JIRAFilter],
        account: int,
        mdb: Database,
        cache: aiomcache.Client | None,
    ) -> _ReleaseJIRAFiltersHelper:
        filter_ = JIRAFilter.combine(*jiras)
        if len(jiras) == 1:
            with_jira = JIRAEntityToFetch.NOTHING
            grouper = _global_grouper
        else:
            with_jira = _get_jira_entities_to_fetch(jiras, ())
            filters_to_grouping = await _JIRAFilterToGroupingConverter.build(
                jiras, account, mdb, cache,
            )
            grouper = partial(group_release_facts_by_jira, filters_to_grouping(jiras))
        return cls(filter_, with_jira, grouper)


def _get_jira_entities_to_fetch(
    jira_filters: Iterable[JIRAFilter],
    pr_metrics: Iterable[str],
) -> JIRAEntityToFetch:
    """Get entities to fetch, in order to compute `pr_metrics` and then group by the filter."""
    jira_entities_to_fetch = JIRAEntityToFetch.NOTHING
    if pr_metrics_need_jira_mapping(pr_metrics):
        jira_entities_to_fetch |= JIRAEntityToFetch.ISSUES
    # for projects, types, and priorities, if at least one filter constraints on the property
    # we need to fetch it to group on it
    for jira_filter in jira_filters:
        if jira_filter.projects:
            jira_entities_to_fetch |= JIRAEntityToFetch.PROJECTS
        if jira_filter.issue_types:
            jira_entities_to_fetch |= JIRAEntityToFetch.TYPES
        if jira_filter.priorities:
            jira_entities_to_fetch |= JIRAEntityToFetch.PRIORITIES
        if jira_filter.labels.include:
            jira_entities_to_fetch |= JIRAEntityToFetch.LABELS
    return jira_entities_to_fetch


def _compose_cache_key_repositories(repositories: Sequence[Collection[str]]) -> str:
    return ",".join(str(sorted(r)) for r in repositories)


AnyParticipants = PRParticipants | ReleaseParticipants | JIRAParticipants


def _compose_cache_key_participants(participants: Sequence[AnyParticipants]) -> str:
    return ";".join(
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(p.items())) for p in participants
    )


def _intersect_items_groups(
    n_groups: int,
    n_items: int,
    *group_arrays: np.ndarray,
) -> list[np.ndarray]:
    """Intersect lists of groups.

    Each group in the result will be the intersection of groups in `group_arrays` index by index.
    Each group_arrays must generate `n_groups`.

    """
    groups = np.empty(n_groups, dtype=object)
    group = np.empty(n_items, dtype=np.uint8)
    for i, group_masks in enumerate(zip(*group_arrays)):
        group[:] = 0
        for mask in group_masks:
            group[mask] += 1
        groups[i] = np.flatnonzero(group == len(group_masks))
    return groups


def _global_grouper(facts: pd.Dataframe) -> list[npt.NDArray[int]]:
    """Group the facts to select everything, in a single group."""
    return [np.arange(len(facts))]


def make_calculator(
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> MetricEntriesCalculator:
    """Get the metrics calculator according to the account's features."""
    return MetricEntriesCalculator(
        account, meta_ids, DEFAULT_QUANTILE_STRIDE, mdb, pdb, rdb, cache,
    )


@dataclass(frozen=True, slots=True)
class MetricsLineRequest:
    """Common base for multiple metrics request classes."""

    metrics: Sequence[str]
    time_intervals: Sequence[Sequence[datetime]]
    teams: Sequence[TeamSpecificFilters]
    metric_params: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        """Summarize the request."""
        return (
            f"<[{', '.join(self.metrics)}],"
            f" [{', '.join('[%s]' % ', '.join(str(d) for d in s) for s in self.time_intervals)}],"
            f" [{', '.join(str(f) for f in self.teams)}]>"
        )

    def __sentry_repr__(self) -> str:
        """Format for Sentry reports."""
        return str(self)

    def all_jira_filters(self) -> Iterator[JIRAFilter]:
        """Collect all JIRAFilter from the included TeamSpecificFilters."""
        return (td.jira_filter for td in self.teams)


@dataclass(frozen=True, slots=True)
class TeamSpecificFilters:
    """Filters that are different for each team."""

    team_id: int
    # filters start here
    repositories: tuple[str]
    participants: PRParticipants | ReleaseParticipants | JIRAParticipants
    jira_filter: JIRAFilter
    # add more filters here

    def __str__(self) -> str:
        """Format the filters as a stable string."""
        # participants are fully defined by the team ID
        return f"{self.team_id}|{','.join(self.repositories)}|{self.jira_filter}"

    def __sentry_repr__(self):
        """Format for Sentry."""
        dikt = dataclasses.asdict(self)
        del dikt["participants"]  # fully defined by the team ID
        return str(dikt)


@dataclass(slots=True)
class MinePullRequestMetrics:
    """Various statistics about mined pull requests."""

    count: int
    done_count: int
    merged_count: int
    open_count: int
    undead_count: int

    @classmethod
    def empty(cls) -> MinePullRequestMetrics:
        """Initialize a new MinePullRequestMetrics instance filled with zeros."""
        return MinePullRequestMetrics(0, 0, 0, 0, 0)

    def as_db(self) -> Iterator[HealthMetric]:
        """Generate HealthMetric-s from this instance."""
        yield HealthMetric(name="prs_count", value=self.count)
        yield HealthMetric(name="prs_done_count", value=self.done_count)
        yield HealthMetric(name="prs_merged_count", value=self.merged_count)
        yield HealthMetric(name="prs_open_count", value=self.open_count)
        yield HealthMetric(name="prs_undead_count", value=self.undead_count)


class PRFactsCalculator:
    """Calculator for Pull Requests facts."""

    _log = logging.getLogger(f"{metadata.__package__}.PRFactsCalculator")

    def __init__(
        self,
        account: int,
        meta_ids: tuple[int, ...],
        mdb: Database,
        pdb: Database,
        rdb: Database,
        pr_miner: Type[PullRequestMiner] = PullRequestMiner,
        branch_miner: Type[BranchMiner] = BranchMiner,
        done_prs_facts_loader: Type[DonePRFactsLoader] = DonePRFactsLoader,
        unfresh_pr_facts_fetcher: Type[
            UnfreshPullRequestFactsFetcher
        ] = UnfreshPullRequestFactsFetcher,
        pr_jira_mapper: Type[PullRequestJiraMapper] = PullRequestJiraMapper,
        cache: Optional[aiomcache.Client] = None,
    ):
        """Init the `PRFactsCalculator`."""
        self._account = account
        self._meta_ids = meta_ids
        self._mdb = mdb
        self._pdb = pdb
        self._rdb = rdb
        self._pr_miner = pr_miner
        self._branch_miner = branch_miner
        self._done_prs_facts_loader = done_prs_facts_loader
        self._unfresh_pr_facts_fetcher = unfresh_pr_facts_fetcher
        self._pr_jira_mapper = pr_jira_mapper
        self._cache = cache

    async def __call__(
        self,
        time_from: datetime,
        time_to: datetime,
        repositories: set[str],
        participants: PRParticipants,
        labels: LabelFilter,
        jira: JIRAFilter,
        exclude_inactive: bool,
        bots: set[str],
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        fresh: bool,
        with_jira: JIRAEntityToFetch | int,
        branches: Optional[pd.DataFrame] = None,
        default_branches: Optional[dict[str, str]] = None,
        on_prs_known: Optional[Callable[[Collection[int]], Any]] = None,
        metrics: Optional[MinePullRequestMetrics] = None,
    ) -> pd.DataFrame:
        """
        Calculate facts about pull request on GitHub.

        :param exclude_inactive: Do not load PRs without events between `time_from` and `time_to`.
        :param fresh: If the number of done PRs for the time period and filters exceeds \
                      `unfresh_mode_threshold`, force querying mdb instead of pdb only.
        :param with_jira: JIRA information to load for each PR.
        :param on_prs_known: Callback that we invoke as soon as we define the list of PR node IDs \
                             of the facts.
        :return: PullRequestFacts packed in a Pandas DataFrame.
        """
        df, *_, from_scratch = await self._call_pr_facts_calculator_cached(
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
            with_jira,
            branches,
            default_branches,
            on_prs_known,
        )
        if df.empty:
            df = pd.DataFrame(columns=PullRequestFacts.f)
        if on_prs_known is not None and not from_scratch:
            on_prs_known(df[PullRequestFacts.f.node_id].values)
        if metrics is not None:
            self._set_count_metrics(df, metrics)
        return df

    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=serialize_args,
        deserialize=deserialize_args,
        key=lambda time_from, time_to, repositories, participants, labels, jira, exclude_inactive, release_settings, logical_settings, fresh, **_: (  # noqa
            time_from,
            time_to,
            ",".join(sorted(repositories)),
            ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
            labels,
            jira,
            exclude_inactive,
            release_settings,
            logical_settings,
            fresh,
        ),
        postprocess=_postprocess_cached_facts,
        cache=lambda self, **_: self._cache,
    )
    async def _call_pr_facts_calculator_cached(
        self,
        time_from: datetime,
        time_to: datetime,
        repositories: set[str],
        participants: PRParticipants,
        labels: LabelFilter,
        jira: JIRAFilter,
        exclude_inactive: bool,
        bots: set[str],
        release_settings: ReleaseSettings,
        logical_settings: LogicalRepositorySettings,
        prefixer: Prefixer,
        fresh: bool,
        with_jira: JIRAEntityToFetch | int,
        branches: Optional[pd.DataFrame],
        default_branches: Optional[dict[str, str]],
        on_prs_known: Optional[Callable[[Collection[int]], Any]],
    ) -> tuple[pd.DataFrame, JIRAEntityToFetch | int, bool]:
        assert isinstance(repositories, set)
        if branches is None or default_branches is None:
            branches, default_branches = await self._branch_miner.load_branches(
                repositories,
                prefixer,
                self._account,
                self._meta_ids,
                self._mdb,
                self._pdb,
                self._cache,
            )
        precomputed_tasks = [
            self._done_prs_facts_loader.load_precomputed_done_facts_filters(
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
                self._done_prs_facts_loader.load_precomputed_done_candidates(
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
        (precomputed_facts, ambiguous), *blacklist = await gather(
            *precomputed_tasks, op="precomputed_tasks",
        )
        precomputed_node_ids = np.unique(
            np.fromiter(
                (node_id for node_id, _ in precomputed_facts), int, len(precomputed_facts),
            ),
        )
        if blacklist:
            blacklist = (sorted_union1d(blacklist[0][0], precomputed_node_ids), blacklist[0][1])
        else:
            blacklist = (precomputed_node_ids, ambiguous)
        add_pdb_hits(self._pdb, "load_precomputed_done_facts_filters", len(precomputed_facts))

        prpk = PRParticipationKind
        if (
            (
                len(precomputed_facts) > unfresh_prs_threshold
                or len(participants.get(prpk.AUTHOR, [])) > unfresh_participants_threshold
                or len(branches) > unfresh_branches_threshold
            )
            and not fresh
            and not (participants.keys() - {prpk.AUTHOR, prpk.MERGER})
        ):
            facts = await self._unfresh_pr_facts_fetcher.fetch_pull_request_facts_unfresh(
                self._pr_miner,
                precomputed_facts,
                ambiguous,
                time_from,
                time_to,
                repositories,
                participants,
                labels,
                jira,
                self._pr_jira_mapper if with_jira else None,
                with_jira,
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
                on_prs_known=on_prs_known,
                done_node_ids=precomputed_node_ids,
            )
            return df_from_structs(facts.values()), with_jira, True

        if with_jira:
            # schedule loading the PR->JIRA mapping
            done_jira_map_task = asyncio.create_task(
                self._pr_jira_mapper.load(
                    precomputed_node_ids, with_jira, self._meta_ids, self._mdb,
                ),
                name="load_pr_jira_mapping/done",
            )
        done_deployments_task = asyncio.create_task(
            self._pr_miner.fetch_pr_deployments(
                precomputed_node_ids, self._account, self._pdb, self._rdb,
            ),
            name="fetch_pr_deployments/done",
        )
        add_pdb_misses(self._pdb, "fresh", 1)
        date_from, date_to = coarsen_time_interval(time_from, time_to)
        # the adjacent out-of-range pieces [date_from, time_from] and [time_to, date_to]
        # are effectively discarded later in BinnedMetricCalculator
        tasks = [
            self._pr_miner.mine(
                date_from,
                date_to,
                time_from,
                time_to,
                repositories,
                participants,
                labels,
                jira,
                with_jira,
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
                self._pr_miner.filter_jira(
                    precomputed_node_ids,
                    jira,
                    self._meta_ids,
                    self._mdb,
                    self._cache,
                    model=NodePullRequest,
                    columns=[NodePullRequest.node_id],
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

        if on_prs_known is not None:
            on_prs_known(
                sorted_union1d(
                    precomputed_node_ids, np.unique(miner.dfs.prs.index.get_level_values(0)),
                ),
            )
        new_deps = miner.dfs.deployments.copy()

        # used later to retrieve jira mapping for mined prs
        if with_jira:
            columns = JIRAEntityToFetch.to_columns(with_jira & ~JIRAEntityToFetch.ISSUES)
            new_jira = miner.dfs.jiras[[c.name for c in columns]].copy(deep=False)
            new_jira.index = new_jira.index.copy()
            if with_jira & JIRAEntityToFetch.ISSUES:
                new_jira[
                    JIRAEntityToFetch.to_columns(JIRAEntityToFetch.ISSUES)[0].name
                ] = new_jira.index.get_level_values(1).values

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
                    time_to,
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
                    time_to,
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
        if with_jira:
            tasks.append(done_jira_map_task)
        done_deps, *done_jira_map = await gather(*tasks)
        if with_jira:
            jira_map = done_jira_map[0]
            self._pr_jira_mapper.append_from_df(jira_map, new_jira)
            self._pr_jira_mapper.apply_to_pr_facts(precomputed_facts, jira_map)
        else:
            self._pr_jira_mapper.apply_empty_to_pr_facts(precomputed_facts)

        # TODO: miner returns in dfs.deployments some PRs included in blacklist,
        # so some PRs can be both in done_deps and new_deps, find out why
        dupl_new_deps = np.intersect1d(
            new_deps.index.values, done_deps.index.values, assume_unique=True,
        )
        if dupl_new_deps.size > 0:
            new_deps.drop(dupl_new_deps, inplace=True)

        self._unfresh_pr_facts_fetcher.append_deployments(
            precomputed_facts, pd.concat([done_deps, new_deps]), self._log,
        )
        all_facts_iter = chain(precomputed_facts.values(), mined_facts.values())
        all_facts_df = df_from_structs(
            all_facts_iter, length=len(precomputed_facts) + len(mined_facts),
        )
        return all_facts_df, with_jira, True

    @staticmethod
    def _set_count_metrics(facts: pd.DataFrame, metrics: MinePullRequestMetrics) -> None:
        metrics.count = len(facts)
        metrics.done_count = int(facts[PullRequestFacts.f.done].sum())
        metrics.merged_count = int(
            (facts[PullRequestFacts.f.merged].notnull() & ~facts[PullRequestFacts.f.done]).sum(),
        )
        metrics.open_count = int(facts[PullRequestFacts.f.closed].isnull().sum())
        metrics.undead_count = int(facts[PullRequestFacts.f.force_push_dropped].sum())


class ParticipantsMerge:
    """Utilities to merge multiple collections of participants."""

    @staticmethod
    def pr(participants: Iterable[PRParticipants]) -> PRParticipants:
        """Merge a set of `PRParticipants`."""
        all_participants: dict[PRParticipationKind, set[str]] = {}
        for p in participants:
            for k, v in p.items():
                all_participants.setdefault(k, set()).update(v)
        return all_participants

    @staticmethod
    def release(participants: Iterable[ReleaseParticipants]) -> ReleaseParticipants:
        """Merge a set of `ReleaseParticipants`."""
        merged: dict[ReleaseParticipationKind, set[int]] = {}
        for dikt in participants:
            for k, v in dikt.items():
                merged.setdefault(k, set()).update(v)
        return {k: list(v) for k, v in merged.items()}

    @staticmethod
    def jira(
        participants: Collection[JIRAParticipants],
    ) -> tuple[Collection[str], Collection[str], Collection[str]]:
        """Merge a collection of `JIRAParticipants`."""
        reporters = list(
            set(
                chain.from_iterable(
                    [p.lower() for p in g.get(JIRAParticipationKind.REPORTER, [])]
                    for g in participants
                ),
            ),
        )
        assignees = list(
            set(
                chain.from_iterable(
                    [
                        (p.lower() if p is not None else None)
                        for p in g.get(JIRAParticipationKind.ASSIGNEE, [])
                    ]
                    for g in participants
                ),
            ),
        )
        commenters = list(
            set(
                chain.from_iterable(
                    [p.lower() for p in g.get(JIRAParticipationKind.COMMENTER, [])]
                    for g in participants
                ),
            ),
        )
        return reporters, assignees, commenters
