import asyncio
from collections import defaultdict, namedtuple
from dataclasses import dataclass, fields as dataclass_fields
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from itertools import chain, repeat
import logging
import pickle
from typing import Collection, Dict, Generator, Iterable, Iterator, KeysView, List, \
    Mapping, Optional, Sequence, Set, Tuple, Union

import aiomcache
import numpy as np
import pandas as pd
from pandas.core.common import flatten
from sqlalchemy import sql
from sqlalchemy.orm import aliased
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached, CancelCache, short_term_exptime
from athenian.api.controllers.logical_repos import coerce_logical_repos
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.commit import BRANCH_FETCH_COMMITS_COLUMNS, \
    DAG, fetch_precomputed_commit_history_dags, fetch_repository_commits_no_branch_dates
from athenian.api.controllers.miners.github.dag_accelerated import searchsorted_inrange
from athenian.api.controllers.miners.github.label import fetch_labels_to_filter
from athenian.api.controllers.miners.github.logical import split_logical_prs
from athenian.api.controllers.miners.github.precomputed_prs import \
    discover_inactive_merged_unreleased_prs, MergedPRFactsLoader, OpenPRFactsLoader, \
    update_unreleased_prs
from athenian.api.controllers.miners.github.release_load import ReleaseLoader
from athenian.api.controllers.miners.github.release_match import PullRequestToReleaseMapper, \
    ReleaseToPullRequestMapper
from athenian.api.controllers.miners.github.released_pr import matched_by_column
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.controllers.miners.types import DeploymentConclusion, MinedPullRequest, \
    nonemax, nonemin, PRParticipants, PRParticipationKind, PullRequestFacts, PullRequestFactsMap
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.settings import LogicalRepositorySettings, ReleaseMatch, \
    ReleaseSettings
from athenian.api.db import add_pdb_misses, Database, DatabaseLike
from athenian.api.defer import AllEvents, defer
from athenian.api.int_to_str import int_to_str
from athenian.api.models.metadata.github import Base, NodePullRequestJiraIssues, \
    PullRequest, PullRequestComment, PullRequestCommit, PullRequestLabel, PullRequestReview, \
    PullRequestReviewComment, PullRequestReviewRequest, PushCommit, Release
from athenian.api.models.metadata.jira import Component, Issue
from athenian.api.models.persistentdata.models import DeploymentNotification
from athenian.api.models.precomputed.models import GitHubPullRequestDeployment
from athenian.api.tracing import sentry_span


@dataclass
class PRDataFrames(Mapping[str, pd.DataFrame]):
    """Set of dataframes with all the PR data we can reach."""

    prs: pd.DataFrame
    commits: pd.DataFrame
    releases: pd.DataFrame
    jiras: pd.DataFrame
    reviews: pd.DataFrame
    review_comments: pd.DataFrame
    review_requests: pd.DataFrame
    comments: pd.DataFrame
    labels: pd.DataFrame
    deployments: pd.DataFrame

    def __iter__(self) -> Iterator[str]:
        """Implement iter() - return an iterator over the field names."""
        return iter((f.name for f in dataclass_fields(self)))

    def __getitem__(self, key: str) -> pd.DataFrame:
        """Implement self[key]."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: pd.DataFrame) -> None:
        """Implement self[key] = value."""
        for f in dataclass_fields(self):
            if key == f.name:
                break
        else:
            raise KeyError(key)
        setattr(self, key, value)

    def __len__(self) -> int:
        """Implement len()."""
        return len(dataclass_fields(self))


class PullRequestMiner:
    """Load all the information related to Pull Requests from the metadata DB. Iterate over it \
    to access individual PR objects."""

    CACHE_TTL = short_term_exptime
    log = logging.getLogger("%s.PullRequestMiner" % metadata.__package__)
    ReleaseMappers = namedtuple("ReleaseMappers", [
        "map_releases_to_prs", "map_prs_to_releases", "load_releases"])
    mappers = ReleaseMappers(
        map_releases_to_prs=ReleaseToPullRequestMapper.map_releases_to_prs,
        map_prs_to_releases=PullRequestToReleaseMapper.map_prs_to_releases,
        load_releases=ReleaseLoader.load_releases,
    )

    def __init__(self, dfs: PRDataFrames):
        """Initialize a new instance of `PullRequestMiner`."""
        self._dfs = dfs

    @property
    def dfs(self) -> PRDataFrames:
        """Return the bound dataframes with PR information."""
        return self._dfs

    def __len__(self) -> int:
        """Return the number of loaded pull requests."""
        return len(self._dfs.prs)

    def __iter__(self) -> Generator[MinedPullRequest, None, None]:
        """Iterate over the individual pull requests."""
        assert self._dfs.prs.index.nlevels == 2
        df_fields = [f.name for f in dataclass_fields(MinedPullRequest) if f.name != "pr"]
        dfs = []
        grouped_df_iters = []
        index_backup = []
        for k in df_fields:
            plural = k.endswith("s")
            df = getattr(self._dfs, k if plural else (k + "s"))  # type: pd.DataFrame
            dfs.append(df)
            # our very own groupby() allows us to call take() with reduced overhead
            node_ids = df.index.get_level_values(0).values.astype(int, copy=False)
            with_repos = k in ("release", "deployments")
            if df.index.nlevels > 1:
                # the second level adds determinism to the iteration order
                second_level = df.index.get_level_values(1).values
                node_ids_bytes = int_to_str(node_ids)
                if second_level.dtype == int:
                    order_keys = np.char.add(node_ids_bytes, int_to_str(second_level))
                else:
                    order_keys = np.char.add(node_ids_bytes,
                                             second_level.astype("S", copy=False))
            else:
                order_keys = node_ids
            df_order = np.argsort(order_keys)
            if not with_repos:
                unique_node_ids, node_ids_unique_counts = np.unique(node_ids, return_counts=True)
                offsets = np.zeros(len(node_ids_unique_counts) + 1, dtype=int)
                np.cumsum(node_ids_unique_counts, out=offsets[1:])
                groups = self._iter_by_split(df_order, offsets)
                grouped_df_iters.append(iter(zip(unique_node_ids, repeat(None), groups)))
            else:
                _, unique_counts = np.unique(order_keys, return_counts=True)
                node_ids = node_ids[df_order]
                repos = df.index.get_level_values(1).values[df_order].astype("U")
                offsets = np.zeros(len(unique_counts) + 1, dtype=int)
                np.cumsum(unique_counts, out=offsets[1:])
                groups = self._iter_by_split(df_order, offsets)
                grouped_df_iters.append(iter(zip(
                    node_ids[offsets[:-1]], repos[offsets[:-1]], groups)))
            if plural:
                index_backup.append(df.index)
                df.index = df.index.droplevel(0)
            else:
                index_backup.append(None)
        try:
            grouped_df_states = []
            for i in grouped_df_iters:
                try:
                    grouped_df_states.append(next(i))
                except StopIteration:
                    grouped_df_states.append((None, None, None))
            empty_df_cache = {}
            pr_columns = [PullRequest.node_id.name, PullRequest.repository_full_name.name]
            pr_columns.extend(self._dfs.prs.columns)
            if not self._dfs.prs.index.is_monotonic_increasing:
                raise IndexError("PRs index must be pre-sorted ascending: "
                                 "prs.sort_index(inplace=True)")
            for pr_tuple in self._dfs.prs.itertuples():
                (pr_node_id, repo), *pr_tuple = pr_tuple
                items = {"pr": dict(zip(pr_columns, [pr_node_id, repo] + pr_tuple))}
                for i, (k, (state_pr_node_id, state_repo, gdf), git, df) in enumerate(zip(
                        df_fields, grouped_df_states, grouped_df_iters, dfs)):
                    while state_pr_node_id is not None and (
                            state_pr_node_id < pr_node_id
                            or (state_pr_node_id == pr_node_id
                                and state_repo is not None
                                and state_repo < repo)):
                        try:
                            state_pr_node_id, state_repo, gdf = next(git)
                        except StopIteration:
                            state_pr_node_id, state_repo, gdf = None, None, None
                    grouped_df_states[i] = state_pr_node_id, state_repo, gdf
                    if state_pr_node_id == pr_node_id and \
                            (state_repo is None or state_repo == repo):
                        if not k.endswith("s"):
                            # much faster than items.iloc[gdf[0]]
                            gdf = {c: v for c, v in zip(df.columns, df._data.fast_xs(gdf[0]))}
                        else:
                            gdf = df.take(gdf)
                        items[k] = gdf
                    else:
                        try:
                            items[k] = empty_df_cache[k]
                        except KeyError:
                            if k.endswith("s"):
                                empty_val = df.iloc[:0].copy()
                            else:
                                empty_val = {c: None for c in df.columns}
                            items[k] = empty_df_cache[k] = empty_val
                yield MinedPullRequest(**items)
        finally:
            for df, index in zip(dfs, index_backup):
                if index is not None:
                    df.index = index

    def drop(self, node_ids: Collection[int]) -> pd.Index:
        """
        Remove PRs from the given collection of PR node IDs in-place.

        Node IDs don't have to be all present.

        :return: Actually removed node IDs.
        """
        removed = self._dfs.prs.index.get_level_values(0).intersection(node_ids)
        if removed.empty:
            return removed
        self._dfs.prs.drop(removed, inplace=True)
        for df in self._dfs.values():
            df.drop(removed, inplace=True, errors="ignore",
                    level=0 if isinstance(df.index, pd.MultiIndex) else None)
        return removed

    def _deserialize_mine_cache(buffer: bytes) -> Tuple[PRDataFrames,
                                                        PullRequestFactsMap,
                                                        Set[str],
                                                        PRParticipants,
                                                        LabelFilter,
                                                        JIRAFilter,
                                                        Dict[str, ReleaseMatch],
                                                        asyncio.Event]:
        stuff = pickle.loads(buffer)
        event = asyncio.Event()
        event.set()
        return (*stuff, event)

    @sentry_span
    def _postprocess_cached_prs(
            result: Tuple[PRDataFrames,
                          PullRequestFactsMap,
                          Set[str],
                          PRParticipants,
                          LabelFilter,
                          JIRAFilter,
                          bool,
                          Dict[str, ReleaseMatch],
                          asyncio.Event],
            date_to: date,
            repositories: Set[str],
            participants: PRParticipants,
            labels: LabelFilter,
            jira: JIRAFilter,
            with_jira_map: bool,
            pr_blacklist: Optional[Tuple[Collection[int], Dict[str, List[int]]]],
            truncate: bool,
            **_) -> Tuple[PRDataFrames,
                          PullRequestFactsMap,
                          Set[str],
                          PRParticipants,
                          LabelFilter,
                          JIRAFilter,
                          bool,
                          Dict[str, ReleaseMatch],
                          asyncio.Event]:
        dfs, _, cached_repositories, cached_participants, cached_labels, cached_jira, \
            cached_with_jira_map, _, _ = result
        if with_jira_map and not cached_with_jira_map:
            raise CancelCache()
        cls = PullRequestMiner
        if (repositories - cached_repositories or
                not cls._check_participants_compatibility(cached_participants, participants) or
                not cached_labels.compatible_with(labels) or
                not cached_jira.compatible_with(jira)):
            raise CancelCache()
        to_remove = set()
        if pr_blacklist is not None:
            to_remove.update(pr_blacklist[0])
        if no_logical_repos := (coerce_logical_repos(repositories).keys() == repositories):
            to_remove.update(dfs.prs.index.get_level_values(0).values[
                np.in1d(dfs.prs.index.get_level_values(1).values,
                        list(repositories), assume_unique=True, invert=True),
            ])
        time_to = None if truncate else pd.Timestamp(date_to, tzinfo=timezone.utc)
        to_remove.update(cls._find_drop_by_participants(dfs, participants, time_to))
        to_remove.update(cls._find_drop_by_labels(dfs, labels))
        to_remove.update(cls._find_drop_by_jira(dfs, jira))
        cls._drop(dfs, to_remove)
        if not no_logical_repos:
            dfs.prs = dfs.prs.take(np.flatnonzero(
                np.in1d(dfs.prs.index.get_level_values(1).values,
                        list(repositories), assume_unique=True),
            ))
        return result

    @classmethod
    @sentry_span
    @cached(
        exptime=lambda cls, **_: cls.CACHE_TTL,
        serialize=lambda r: pickle.dumps(r[:-1]),
        deserialize=_deserialize_mine_cache,
        key=lambda date_from, date_to, exclude_inactive, release_settings, logical_settings, updated_min, updated_max, pr_blacklist, truncate, **_: (  # noqa
            date_from.toordinal(), date_to.toordinal(), exclude_inactive,
            release_settings, logical_settings,
            updated_min.timestamp() if updated_min is not None else None,
            updated_max.timestamp() if updated_max is not None else None,
            ",".join(map(str, sorted(pr_blacklist[0]) if pr_blacklist is not None else [])),
            truncate,
        ),
        postprocess=_postprocess_cached_prs,
    )
    async def _mine(cls,
                    date_from: date,
                    date_to: date,
                    repositories: Set[str],
                    participants: PRParticipants,
                    labels: LabelFilter,
                    jira: JIRAFilter,
                    with_jira_map: bool,
                    branches: pd.DataFrame,
                    default_branches: Dict[str, str],
                    exclude_inactive: bool,
                    release_settings: ReleaseSettings,
                    logical_settings: LogicalRepositorySettings,
                    updated_min: Optional[datetime],
                    updated_max: Optional[datetime],
                    pr_blacklist: Optional[Tuple[Collection[int], Dict[str, List[int]]]],
                    truncate: bool,
                    prefixer: Prefixer,
                    account: int,
                    meta_ids: Tuple[int, ...],
                    mdb: Database,
                    pdb: Database,
                    rdb: Database,
                    cache: Optional[aiomcache.Client],
                    ) -> Tuple[PRDataFrames,
                               PullRequestFactsMap,
                               Set[str],
                               PRParticipants,
                               LabelFilter,
                               JIRAFilter,
                               bool,
                               Dict[str, ReleaseMatch],
                               asyncio.Event]:
        assert isinstance(date_from, date) and not isinstance(date_from, datetime)
        assert isinstance(date_to, date) and not isinstance(date_to, datetime)
        assert isinstance(repositories, set)
        assert isinstance(mdb, Database)
        assert isinstance(pdb, Database)
        assert isinstance(rdb, Database)
        assert (updated_min is None) == (updated_max is None)
        time_from, time_to = (pd.Timestamp(t, tzinfo=timezone.utc) for t in (date_from, date_to))
        pr_blacklist_expr = ambiguous = None
        if pr_blacklist is not None:
            pr_blacklist, ambiguous = pr_blacklist
            if len(pr_blacklist) > 0:
                pr_blacklist_expr = PullRequest.node_id.notin_any_values(pr_blacklist)
        if logical_settings.has_logical_prs():
            physical_repos = coerce_logical_repos(repositories).keys()
        else:
            physical_repos = repositories
        pdags = await fetch_precomputed_commit_history_dags(physical_repos, account, pdb, cache)
        fetch_branch_dags_task = asyncio.create_task(
            cls._fetch_branch_dags(
                physical_repos, pdags, branches, account, meta_ids, mdb, pdb, cache),
            name="_fetch_branch_dags",
        )
        # the heaviest task should always go first
        tasks = [
            cls.mappers.map_releases_to_prs(
                repositories, branches, default_branches, time_from, time_to,
                participants.get(PRParticipationKind.AUTHOR, []),
                participants.get(PRParticipationKind.MERGER, []),
                jira, release_settings, logical_settings, updated_min, updated_max, pdags,
                prefixer, account, meta_ids, mdb, pdb, rdb, cache, pr_blacklist_expr, None,
                truncate=truncate),
            cls.fetch_prs(
                time_from, time_to, physical_repos, participants, labels, jira,
                exclude_inactive, pr_blacklist_expr, None, branches, pdags, account, meta_ids,
                mdb, pdb, cache, updated_min=updated_min, updated_max=updated_max,
                fetch_branch_dags_task=fetch_branch_dags_task),
            cls.map_deployments_to_prs(
                physical_repos, time_from, time_to, participants,
                labels, jira, updated_min, updated_max, branches, pdags,
                account, meta_ids, mdb, pdb, cache, pr_blacklist,
                fetch_branch_dags_task=fetch_branch_dags_task),
        ]
        # the following is a very rough approximation regarding updated_min/max:
        # we load all of none of the inactive merged PRs
        # see also: load_precomputed_done_candidates() which generates `ambiguous`
        if not exclude_inactive and (updated_min is None or updated_min <= time_from):
            tasks.append(cls._fetch_inactive_merged_unreleased_prs(
                time_from, time_to, repositories, participants, labels, jira, default_branches,
                release_settings, logical_settings.has_logical_prs(),
                prefixer, account, meta_ids, mdb, pdb, cache))
            # we don't load inactive released undeployed PRs because nobody needs them
        (
            (released_prs, releases, release_settings, matched_bys,
             release_dags, precomputed_observed),
            (prs, branch_dags, _),
            deployed_prs,
            *unreleased,
        ) = await gather(*tasks)
        del pr_blacklist_expr
        deployed_releases_task = None
        if not deployed_prs.empty:
            covered_prs = prs.index.union(released_prs.index)
            if unreleased:
                covered_prs = covered_prs.union(unreleased[0].index)
            new_prs = deployed_prs.index.difference(covered_prs)
            if not new_prs.empty:
                new_prs = deployed_prs[[
                    PullRequest.merged_at.name, PullRequest.repository_full_name.name,
                ]].loc[new_prs]
                min_deployed_merged = new_prs[PullRequest.merged_at.name].min()
                if min_deployed_merged < time_from:
                    deployed_releases_task = asyncio.create_task(
                        cls.mappers.load_releases(
                            new_prs[PullRequest.repository_full_name.name].unique(),
                            branches, default_branches, min_deployed_merged, time_from,
                            release_settings, logical_settings, prefixer, account, meta_ids,
                            mdb, pdb, rdb, cache),
                        name="PullRequestMiner.mine/deployed_releases",
                    )
        concatenated = [prs, released_prs, deployed_prs, *unreleased]
        missed_prs = cls._extract_missed_prs(ambiguous, pr_blacklist, deployed_prs, matched_bys)
        if missed_prs:
            add_pdb_misses(pdb, "PullRequestMiner.mine/blacklist",
                           sum(len(v) for v in missed_prs.values()))
            # these PRs are released by branch and not by tag, and we require by tag.
            # we have not fetched them yet because they are in pr_blacklist
            # and they are in pr_blacklist because we have previously loaded them in
            # load_precomputed_done_candidates();
            # now fetch only these `missed_prs`, respecting the filters.
            pr_whitelist = PullRequest.node_id.in_(
                list(chain.from_iterable(missed_prs.values())))
            tasks = [
                cls.mappers.map_releases_to_prs(
                    missed_prs, branches, default_branches, time_from, time_to,
                    participants.get(PRParticipationKind.AUTHOR, []),
                    participants.get(PRParticipationKind.MERGER, []),
                    jira, release_settings, logical_settings, updated_min, updated_max, pdags,
                    prefixer, account, meta_ids, mdb, pdb, rdb, cache, None, pr_whitelist,
                    truncate, precomputed_observed=precomputed_observed),
                cls.fetch_prs(
                    time_from, time_to, missed_prs.keys(), participants, labels, jira,
                    exclude_inactive, None, pr_whitelist, branches, branch_dags, account, meta_ids,
                    mdb, pdb, cache, updated_min=updated_min, updated_max=updated_max,
                    fetch_branch_dags_task=fetch_branch_dags_task),
            ]
            missed_released_prs, (missed_prs, *_) = await gather(*tasks)
            concatenated.extend([missed_released_prs, missed_prs])
        fetch_branch_dags_task.cancel()  # 99.999% that it was awaited, but still
        prs = pd.concat(concatenated, copy=False)
        prs.reset_index(inplace=True)
        prs.drop_duplicates([PullRequest.node_id.name, PullRequest.repository_full_name.name],
                            inplace=True)
        prs.set_index(PullRequest.node_id.name, inplace=True)
        prs.sort_index(inplace=True)

        if unreleased:
            unreleased = np.array([
                unreleased[0].index.values,
                unreleased[0][PullRequest.repository_full_name.name].values,
            ], dtype=object).T
        tasks = [
            # bypass the useless inner caching by calling _mine_by_ids directly
            cls._mine_by_ids(
                prs, unreleased, repositories, time_to, releases, matched_bys,
                branches, default_branches, release_dags, release_settings, logical_settings,
                prefixer, account, meta_ids, mdb, pdb, rdb, cache,
                truncate=truncate, with_jira=with_jira_map,
                extra_releases_task=deployed_releases_task,
                physical_repositories=physical_repos),
            OpenPRFactsLoader.load_open_pull_request_facts(prs, repositories, account, pdb),
        ]
        (dfs, unreleased_facts, unreleased_prs_event), open_facts = await gather(
            *tasks, op="PullRequestMiner.mine/external_data")

        to_drop = cls._find_drop_by_participants(dfs, participants, None if truncate else time_to)
        to_drop |= cls._find_drop_by_labels(dfs, labels)
        if exclude_inactive:
            to_drop |= cls._find_drop_by_inactive(dfs, time_from, time_to)
        cls._drop(dfs, to_drop)

        facts = open_facts
        for k, v in unreleased_facts.items():  # merged unreleased PR precomputed facts
            if v is not None:  # it can be None because the pdb table is filled in two steps
                facts[k] = v

        dfs.prs = split_logical_prs(
            dfs.prs, dfs.labels, repositories, logical_settings)
        return dfs, facts, repositories, participants, labels, jira, with_jira_map, matched_bys, \
            unreleased_prs_event

    _deserialize_mine_cache = staticmethod(_deserialize_mine_cache)
    _postprocess_cached_prs = staticmethod(_postprocess_cached_prs)

    def _deserialize_mine_by_ids_cache(
            buffer: bytes) -> Tuple[PRDataFrames,
                                    PullRequestFactsMap,
                                    asyncio.Event]:
        dfs, facts = pickle.loads(buffer)
        event = asyncio.Event()
        event.set()
        return dfs, facts, event

    @classmethod
    @cached(
        exptime=lambda cls, **_: cls.CACHE_TTL,
        serialize=lambda r: pickle.dumps(r[:-1]),
        deserialize=_deserialize_mine_by_ids_cache,
        key=lambda prs, unreleased, releases, time_to, logical_settings, truncate=True, with_jira=True, **_: (  # noqa
            ",".join(map(str, prs.index.values)),
            ",".join(map(str, unreleased)),
            ",".join(map(str, releases[Release.node_id.name].values)),
            time_to.timestamp(),
            logical_settings,
            truncate,
            with_jira,
        ),
    )
    async def mine_by_ids(cls,
                          prs: pd.DataFrame,
                          unreleased: Collection[Tuple[int, str]],
                          logical_repositories: Union[Set[str], KeysView[str]],
                          time_to: datetime,
                          releases: pd.DataFrame,
                          matched_bys: Dict[str, ReleaseMatch],
                          branches: pd.DataFrame,
                          default_branches: Dict[str, str],
                          dags: Dict[str, DAG],
                          release_settings: ReleaseSettings,
                          logical_settings: LogicalRepositorySettings,
                          prefixer: Prefixer,
                          account: int,
                          meta_ids: Tuple[int, ...],
                          mdb: Database,
                          pdb: Database,
                          rdb: Database,
                          cache: Optional[aiomcache.Client],
                          truncate: bool = True,
                          with_jira: bool = True,
                          physical_repositories: Optional[Union[Set[str], KeysView[str]]] = None,
                          ) -> Tuple[PRDataFrames,
                                     PullRequestFactsMap,
                                     asyncio.Event]:
        """
        Fetch PR metadata for certain PRs.

        :param prs: pandas DataFrame with fetched PullRequest-s. Only the details about those PRs \
                    will be loaded from the DB.
        :param truncate: Do not load anything after `time_to`.
        :param with_jira: Value indicating whether to load the mapped JIRA issues.
        :return: 1. List of mined DataFrame-s. \
                 2. mapping to PullRequestFacts of unreleased merged PRs. \
                 3. Synchronization for updating the pdb table with merged unreleased PRs.
        """
        return await cls._mine_by_ids(
            prs, unreleased, logical_repositories, time_to, releases, matched_bys,
            branches, default_branches, dags, release_settings, logical_settings, prefixer,
            account, meta_ids, mdb, pdb, rdb, cache, truncate=truncate, with_jira=with_jira,
            physical_repositories=physical_repositories)

    _deserialize_mine_by_ids_cache = staticmethod(_deserialize_mine_by_ids_cache)

    @classmethod
    @sentry_span
    async def _mine_by_ids(cls,
                           prs: pd.DataFrame,
                           unreleased: Collection[Tuple[int, str]],
                           logical_repositories: Union[Set[str], KeysView[str]],
                           time_to: datetime,
                           releases: pd.DataFrame,
                           matched_bys: Dict[str, ReleaseMatch],
                           branches: pd.DataFrame,
                           default_branches: Dict[str, str],
                           dags: Dict[str, DAG],
                           release_settings: ReleaseSettings,
                           logical_settings: LogicalRepositorySettings,
                           prefixer: Prefixer,
                           account: int,
                           meta_ids: Tuple[int, ...],
                           mdb: Database,
                           pdb: Database,
                           rdb: Database,
                           cache: Optional[aiomcache.Client],
                           truncate: bool = True,
                           with_jira: bool = True,
                           extra_releases_task: Optional[asyncio.Task] = None,
                           physical_repositories: Optional[Union[Set[str], KeysView[str]]] = None,
                           ) -> Tuple[PRDataFrames,
                                      PullRequestFactsMap,
                                      asyncio.Event]:
        assert prs.index.nlevels == 1
        node_ids = prs.index if len(prs) > 0 else set()
        facts = {}  # precomputed PullRequestFacts about merged unreleased PRs
        unreleased_prs_event: asyncio.Event = None
        merged_unreleased_indexes = []

        @sentry_span
        async def fetch_reviews():
            return await cls._read_filtered_models(
                PullRequestReview, node_ids, time_to, meta_ids, mdb,
                columns=[PullRequestReview.submitted_at, PullRequestReview.state,
                         PullRequestReview.user_login, PullRequestReview.user_node_id],
                created_at=truncate)

        @sentry_span
        async def fetch_review_comments():
            return await cls._read_filtered_models(
                PullRequestReviewComment, node_ids, time_to, meta_ids, mdb,
                columns=[PullRequestReviewComment.created_at, PullRequestReviewComment.user_login,
                         PullRequestReviewComment.user_node_id],
                created_at=truncate)

        @sentry_span
        async def fetch_review_requests():
            return await cls._read_filtered_models(
                PullRequestReviewRequest, node_ids, time_to, meta_ids, mdb,
                columns=[PullRequestReviewRequest.created_at],
                created_at=truncate)

        @sentry_span
        async def fetch_comments():
            return await cls._read_filtered_models(
                PullRequestComment, node_ids, time_to, meta_ids, mdb,
                columns=[PullRequestComment.created_at, PullRequestComment.user_login,
                         PullRequestComment.user_node_id],
                created_at=truncate)

        @sentry_span
        async def fetch_commits():
            return await cls._read_filtered_models(
                PullRequestCommit, node_ids, time_to, meta_ids, mdb,
                columns=[PullRequestCommit.authored_date, PullRequestCommit.committed_date,
                         PullRequestCommit.author_login, PullRequestCommit.committer_login,
                         PullRequestCommit.author_user_id, PullRequestCommit.committer_user_id],
                created_at=truncate)

        @sentry_span
        async def fetch_labels():
            return await cls._read_filtered_models(
                PullRequestLabel, node_ids, time_to, meta_ids, mdb,
                columns=[sql.func.lower(PullRequestLabel.name).label(PullRequestLabel.name.name),
                         PullRequestLabel.description,
                         PullRequestLabel.color],
                created_at=False)

        fetch_labels_task = asyncio.create_task(
            fetch_labels(), name="PullRequestMiner.mine_by_ids/fetch_labels")

        @sentry_span
        async def map_releases():
            anyhow_merged_mask = prs[PullRequest.merged_at.name].notnull().values
            if truncate:
                merged_mask = (prs[PullRequest.merged_at.name] < time_to).values
                nonlocal merged_unreleased_indexes
                merged_unreleased_indexes = np.flatnonzero(anyhow_merged_mask & ~merged_mask)
            else:
                merged_mask = anyhow_merged_mask
            if len(unreleased):
                prs_index = np.char.add(
                    int_to_str(prs.index.values),
                    (prs_repos := prs[PullRequest.repository_full_name.name].values.astype("S")),
                )
                if isinstance(unreleased, np.ndarray):
                    unreleased_index = np.char.add(
                        int_to_str(unreleased[:, 0].astype(int)),
                        unreleased[:, 1].astype(prs_repos.dtype),
                    )
                else:
                    unreleased_index = np.char.add(
                        int_to_str(np.fromiter((p[0] for p in unreleased), int, len(unreleased))),
                        np.array([p[1] for p in unreleased], dtype=prs_repos.dtype),
                    )
                merged_mask &= np.in1d(prs_index, unreleased_index, invert=True)
            merged_prs = prs.take(np.flatnonzero(merged_mask))
            nonlocal releases
            if extra_releases_task is not None:
                await extra_releases_task
                extra_releases, _ = extra_releases_task.result()
                releases = releases.append(extra_releases, ignore_index=True)
            labels = None
            if logical_settings.has_logical_prs():
                nonlocal physical_repositories
                if physical_repositories is None:
                    physical_repositories = coerce_logical_repos(logical_repositories).keys()
                if logical_settings.has_prs_by_label(physical_repositories):
                    await fetch_labels_task
                    labels = fetch_labels_task.result()
                merged_prs = split_logical_prs(
                    merged_prs, labels, logical_repositories, logical_settings)
            else:
                merged_prs = split_logical_prs(merged_prs, None, set(), logical_settings)
            df_facts, other_facts = await gather(
                cls.mappers.map_prs_to_releases(
                    merged_prs, releases, matched_bys, branches, default_branches, time_to,
                    dags, release_settings, prefixer, account, meta_ids, mdb, pdb, cache,
                    labels=labels),
                MergedPRFactsLoader.load_merged_unreleased_pull_request_facts(
                    prs.take(np.flatnonzero(anyhow_merged_mask & ~merged_mask)),
                    nonemax(releases[Release.published_at.name].nonemax(), time_to),
                    LabelFilter.empty(), matched_bys, default_branches, release_settings,
                    prefixer, account, pdb),
            )
            nonlocal facts
            nonlocal unreleased_prs_event
            df, facts, unreleased_prs_event = df_facts
            facts.update(other_facts)
            return df

        async def _fetch_labels():
            await fetch_labels_task
            return fetch_labels_task.result()

        @sentry_span
        async def fetch_jira():
            _map = aliased(NodePullRequestJiraIssues, name="m")
            _issue = aliased(Issue, name="i")
            _issue_epic = aliased(Issue, name="e")
            selected = [
                PullRequest.node_id, _issue.key, _issue.title, _issue.type, _issue.status,
                _issue.created, _issue.updated, _issue.resolved, _issue.labels, _issue.components,
                _issue.acc_id, _issue_epic.key.label("epic"), _issue.project_id,
            ]
            if not with_jira:
                df = pd.DataFrame(columns=[col.name for col in selected
                                           if col not in (_issue.acc_id, _issue.components)])
                df[PullRequest.node_id.name] = df[PullRequest.node_id.name].astype(int)
                return df.set_index([PullRequest.node_id.name, _issue.key.name])
            df = await read_sql_query(
                sql.select(selected).select_from(sql.join(
                    PullRequest, sql.join(
                        _map, sql.join(_issue, _issue_epic, sql.and_(
                            _issue.epic_id == _issue_epic.id,
                            _issue.acc_id == _issue_epic.acc_id), isouter=True),
                        sql.and_(_map.jira_id == _issue.id,
                                 _map.jira_acc == _issue.acc_id)),
                    sql.and_(PullRequest.node_id == _map.node_id,
                             PullRequest.acc_id == _map.node_acc),
                )).where(sql.and_(PullRequest.node_id.in_(node_ids),
                                  PullRequest.acc_id.in_(meta_ids),
                                  _issue.is_deleted.is_(False))),
                mdb, columns=selected, index=[PullRequest.node_id.name, _issue.key.name])
            if df.empty:
                df.drop([Issue.acc_id.name, Issue.components.name], inplace=True, axis=1)
                return df
            components = df[[Issue.acc_id.name, Issue.components.name]] \
                .groupby(Issue.acc_id.name, sort=False).aggregate(lambda s: set(flatten(s)))
            rows = await mdb.fetch_all(
                sql.select([Component.acc_id, Component.id, Component.name])
                .where(sql.or_(*(sql.and_(Component.id.in_(vals),
                                          Component.acc_id == int(acc))
                                 for acc, vals in zip(components.index.values,
                                                      components[Issue.components.name].values)))))
            cmap = {}
            for r in rows:
                cmap.setdefault(r[0], {})[r[1]] = r[2].lower()
            df[Issue.labels.name] = (
                df[Issue.labels.name].apply(lambda i: [s.lower() for s in (i or [])])
                +
                df[[Issue.acc_id.name, Issue.components.name]]
                .apply(lambda row: ([cmap[row[Issue.acc_id.name]][c]
                                    for c in row[Issue.components.name]]
                                    if row[Issue.components.name] is not None else []),
                       axis=1)
            )
            df.drop([Issue.acc_id.name, Issue.components.name], inplace=True, axis=1)
            return df

        # the order is important: it provides the best performance
        # we launch coroutines from the heaviest to the lightest
        dfs = await gather(
            fetch_commits(),
            map_releases(),
            fetch_jira(),
            fetch_reviews(),
            fetch_review_comments(),
            fetch_review_requests(),
            fetch_comments(),
            _fetch_labels(),
            cls.fetch_pr_deployments(node_ids, account, pdb, rdb),
        )
        dfs = PRDataFrames(prs, *dfs)
        if len(merged_unreleased_indexes):
            # if we truncate and there are PRs merged after `time_to`
            merged_unreleased_prs = prs.take(merged_unreleased_indexes)
            label_matches = np.flatnonzero(np.in1d(
                dfs.labels.index.get_level_values(0).values,
                merged_unreleased_prs.index.values))
            labels = {}
            for k, v in zip(dfs.labels.index.get_level_values(0).values[label_matches],
                            dfs.labels[PullRequestLabel.name.name].values[label_matches]):
                try:
                    labels[k].append(v)
                except KeyError:
                    labels[k] = [v]
            other_unreleased_prs_event = asyncio.Event()
            unreleased_prs_event = AllEvents(unreleased_prs_event, other_unreleased_prs_event)
            merged_unreleased_prs = split_logical_prs(
                merged_unreleased_prs, dfs.labels, logical_repositories, logical_settings)
            await defer(update_unreleased_prs(
                merged_unreleased_prs, pd.DataFrame(), time_to, labels, matched_bys,
                default_branches, release_settings, account, pdb, other_unreleased_prs_event),
                "update_unreleased_prs/truncate(%d)" % len(merged_unreleased_indexes))
        return dfs, facts, unreleased_prs_event

    @classmethod
    @sentry_span
    async def mine(cls,
                   date_from: date,
                   date_to: date,
                   time_from: datetime,
                   time_to: datetime,
                   repositories: Set[str],
                   participants: PRParticipants,
                   labels: LabelFilter,
                   jira: JIRAFilter,
                   with_jira_map: bool,
                   branches: pd.DataFrame,
                   default_branches: Dict[str, str],
                   exclude_inactive: bool,
                   release_settings: ReleaseSettings,
                   logical_settings: LogicalRepositorySettings,
                   prefixer: Prefixer,
                   account: int,
                   meta_ids: Tuple[int, ...],
                   mdb: Database,
                   pdb: Database,
                   rdb: Database,
                   cache: Optional[aiomcache.Client],
                   updated_min: Optional[datetime] = None,
                   updated_max: Optional[datetime] = None,
                   pr_blacklist: Optional[Tuple[Collection[int], Dict[str, List[int]]]] = None,
                   truncate: bool = True,
                   ) -> Tuple["PullRequestMiner",
                              PullRequestFactsMap,
                              Dict[str, ReleaseMatch],
                              asyncio.Event]:
        """
        Mine metadata about pull requests according to the numerous filters.

        :param account: State DB account ID.
        :param meta_ids: Metadata (GitHub) account IDs.
        :param date_from: Fetch PRs created starting from this date, inclusive.
        :param date_to: Fetch PRs created ending with this date, inclusive.
        :param time_from: Precise timestamp of since when PR events are allowed to happen.
        :param time_to: Precise timestamp of until when PR events are allowed to happen.
        :param repositories: PRs must belong to these repositories (prefix excluded).
        :param participants: PRs must have these user IDs in the specified participation roles \
                             (OR aggregation). An empty dict means everybody.
        :param labels: PRs must be labeled according to this filter's include & exclude sets.
        :param jira: JIRA filters for those PRs that are matched with JIRA issues.
        :param with_jira_map: Value indicating whether we must load JIRA issues mapped to PRs. \
                              This is independent from filtering PRs by `jira`.
        :param branches: Preloaded DataFrame with branches in the specified repositories.
        :param default_branches: Mapping from repository names to their default branch names.
        :param exclude_inactive: Ors must have at least one event in the given time frame.
        :param release_settings: Release match settings of the account.
        :param logical_settings: Logical repository settings of the account.
        :param updated_min: PRs must have the last update timestamp not older than it.
        :param updated_max: PRs must have the last update timestamp not newer than or equal to it.
        :param mdb: Metadata db instance.
        :param pdb: Precomputed db instance.
        :param rdb: Persistentdata db instance.
        :param cache: memcached client to cache the collected data.
        :param pr_blacklist: completely ignore the existence of these PR node IDs. \
                             The second tuple element is the ambiguous PRs: released by branch \
                             while there were no tag releases and the strategy is `tag_or_branch`.
        :param truncate: activate the "time machine" and erase everything after `time_to`.
        :return: 1. New `PullRequestMiner` with the PRs satisfying to the specified filters. \
                 2. Precomputed facts about unreleased pull requests. \
                    This is an optimization which breaks the abstraction a bit. \
                 3. `matched_bys` - release matches for each repository. \
                 4. Synchronization for updating the pdb table with merged unreleased PRs. \
                    Another abstraction leakage that we have to deal with.
        """
        date_from_with_time = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
        date_to_with_time = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
        assert time_from >= date_from_with_time
        assert time_to <= date_to_with_time
        dfs, facts, _, _, _, _, _, matched_bys, event = await cls._mine(
            date_from, date_to, repositories, participants, labels, jira, with_jira_map, branches,
            default_branches, exclude_inactive, release_settings, logical_settings,
            updated_min, updated_max, pr_blacklist, truncate, prefixer, account, meta_ids,
            mdb, pdb, rdb, cache)
        cls._truncate_prs(dfs, time_from, time_to)
        return cls(dfs), facts, matched_bys, event

    @classmethod
    @sentry_span
    async def fetch_prs(cls,
                        time_from: Optional[datetime],
                        time_to: datetime,
                        repositories: Union[Set[str], KeysView[str]],
                        participants: PRParticipants,
                        labels: LabelFilter,
                        jira: JIRAFilter,
                        exclude_inactive: bool,
                        pr_blacklist: Optional[BinaryExpression],
                        pr_whitelist: Optional[BinaryExpression],
                        branches: pd.DataFrame,
                        dags: Optional[Dict[str, DAG]],
                        account: int,
                        meta_ids: Tuple[int, ...],
                        mdb: Database,
                        pdb: Database,
                        cache: Optional[aiomcache.Client],
                        columns=PullRequest,
                        updated_min: Optional[datetime] = None,
                        updated_max: Optional[datetime] = None,
                        fetch_branch_dags_task: Optional[asyncio.Task] = None,
                        with_labels: bool = False,
                        ) -> Tuple[pd.DataFrame, Dict[str, DAG], Optional[pd.DataFrame]]:
        """
        Query pull requests from mdb that satisfy the given filters.

        Note: we cannot filter by regular PR labels here due to the DB schema limitations,
        so the caller is responsible for fetching PR labels and filtering by them afterward.
        Besides, we cannot filter by participation roles different from AUTHOR and MERGER.

        Note: we cannot load PRs that closed before time_from but released between
        `time_from` and `time_to`. Hence the caller should map_releases_to_prs separately.
        There can be duplicates: PR closed between `time_from` and `time_to` and released
        between `time_from` and `time_to`.

        Note: we cannot load PRs that closed before time_from but deployed between
        `time_from` and `time_to`. Hence the caller should map_deployments_to_prs separately.
        There can be duplicates: PR closed between `time_from` and `time_to` and deployed
        between `time_from` and `time_to`.

        We have to resolve the merge commits of rebased PRs so that they do not appear
        force-push-dropped.

        :return: pandas DataFrame with the PRs indexed by node_id; \
                 commit DAGs that contain the branch heads; \
                 (if was required) DataFrame with PR labels.
        """
        assert isinstance(mdb, Database)
        assert isinstance(pdb, Database)
        pr_list_coro = cls._fetch_prs_by_filters(
            time_from, time_to, repositories, participants, labels, jira, exclude_inactive,
            pr_blacklist, pr_whitelist, meta_ids, mdb, cache, columns=columns,
            updated_min=updated_min, updated_max=updated_max,
        )
        if columns is not PullRequest and PullRequest.merge_commit_id not in columns and \
                PullRequest.merge_commit_sha not in columns:
            prs, labels = await pr_list_coro
            return prs, dags, labels if with_labels else None

        if fetch_branch_dags_task is None:
            fetch_branch_dags_task = cls._fetch_branch_dags(
                repositories, dags, branches, account, meta_ids, mdb, pdb, cache)

        dags, (prs, labels) = await gather(fetch_branch_dags_task, pr_list_coro)

        async def load_labels():
            if not with_labels:
                return None
            if labels is not None:
                return labels
            return await fetch_labels_to_filter(prs.index.values, meta_ids, mdb)

        prs, labels = await gather(
            cls.mark_dead_prs(prs, branches, dags, meta_ids, mdb, columns),
            load_labels(),
        )
        return prs, dags, labels

    @classmethod
    async def mark_dead_prs(cls,
                            prs: pd.DataFrame,
                            branches: pd.DataFrame,
                            dags: Dict[str, DAG],
                            meta_ids: Tuple[int, ...],
                            mdb: Database,
                            columns=PullRequest,
                            ) -> pd.DataFrame:
        """
        Add and fill "dead" column in the `prs` DataFrame.

        A PR is considered dead (force-push-dropped) if it does not exit in the commit DAG and \
        we cannot detect its rebased clone.
        """
        prs["dead"] = False
        if branches.empty:
            return prs
        merged_prs = prs.take(np.nonzero((
            prs[PullRequest.merged_at.name] <= datetime.now(timezone.utc) - timedelta(hours=1)
        ).values)[0])
        # timedelta(hours=1) must match the `exptime` of `fetch_repository_commits()`
        # commits DAGs are cached and may be not fully up to date, so otherwise some PRs may
        # appear as wrongly force push dropped; see also: DEV-554
        if merged_prs.empty:
            return prs
        pr_numbers = merged_prs[PullRequest.number.name].values
        assert merged_prs.index.nlevels == 1
        pr_node_ids = merged_prs.index.values
        pr_repos = merged_prs[PullRequest.repository_full_name.name].values
        repo_order = np.argsort(pr_repos)
        unique_pr_repos, pr_repo_counts = np.unique(pr_repos, return_counts=True)
        pr_merge_hashes = \
            merged_prs[PullRequest.merge_commit_sha.name].values.astype("S40")[repo_order]
        pos = 0
        queries = []
        dead = []
        acc_id_cond = PushCommit.acc_id.in_(meta_ids)
        min_commit_date = merged_prs[PullRequest.merged_at.name].min()
        committed_date_cond = PushCommit.committed_date >= min_commit_date
        substr = sql.func.substr(PushCommit.message, 1, 32)
        sqlite = mdb.url.dialect == "sqlite"
        for repo, n_prs in zip(unique_pr_repos, pr_repo_counts):
            begin_pos = pos
            end_pos = pos + n_prs
            pos += n_prs
            repo_pr_merge_hashes = pr_merge_hashes[begin_pos:end_pos]
            dag_hashes = dags[repo][0]
            if len(dag_hashes) == 0:
                # no branches found in `fetch_repository_commits()`
                continue
            not_found = dag_hashes[
                searchsorted_inrange(dag_hashes, repo_pr_merge_hashes)
            ] != repo_pr_merge_hashes
            indexes = repo_order[begin_pos:end_pos][not_found]
            dead.extend(dead_node_ids := pr_node_ids[indexes])
            repo_cond = PushCommit.repository_full_name == repo
            for pr_node_id, n in zip(dead_node_ids, pr_numbers[indexes]):
                if sqlite:
                    # SQLite does not support parameter recycling
                    acc_id_cond = PushCommit.acc_id.in_(meta_ids)
                    committed_date_cond = PushCommit.committed_date >= min_commit_date
                    substr = sql.func.substr(PushCommit.message, 1, 32)
                    repo_cond = PushCommit.repository_full_name == repo
                queries.append(
                    sql.select([PushCommit.node_id.label("commit_node_id"),
                                PushCommit.sha.label("sha"),
                                sql.literal_column("'" + repo + "'").label("repo"),
                                sql.literal_column(str(pr_node_id)).label("pr_node_id"),
                                PushCommit.committed_date,
                                PushCommit.pushed_date])
                    .where(sql.and_(acc_id_cond,
                                    repo_cond,
                                    committed_date_cond,
                                    substr.like("Merge pull request #%d from %%" % n))))
        if not queries:
            return prs
        prs.loc[dead, "dead"] = True
        # we may have MANY queries here and Postgres responds with StatementTooComplexError
        # split them by 100-sized batches to stay below the resource limits
        batch_size = 100
        tasks = []
        for batch_index in range(0, len(queries), batch_size):
            batch = queries[batch_index:batch_index + batch_size]
            if len(batch) == 1:
                query = batch[0]
            else:
                query = sql.union_all(*batch)
            tasks.append(read_sql_query(query, mdb, [
                "commit_node_id", "sha", "repo", "pr_node_id",
                PushCommit.committed_date, PushCommit.pushed_date,
            ]))
        resolveds = await gather(*tasks, op="mark_dead_prs commit SQL UNION ALL-s")
        resolved = pd.concat(resolveds)
        # look up the candidates in the DAGs
        pr_repos = resolved["repo"].values
        repo_order = np.argsort(pr_repos)
        unique_pr_repos, pr_repo_counts = np.unique(pr_repos, return_counts=True)
        pr_merge_hashes = resolved["sha"].values.astype("S")[repo_order]
        pos = 0
        alive_indexes = []
        for repo, n_prs in zip(unique_pr_repos, pr_repo_counts):
            begin_pos = pos
            end_pos = pos + n_prs
            pos += n_prs
            repo_pr_merge_hashes = pr_merge_hashes[begin_pos:end_pos]
            dag_hashes = dags[repo][0]
            found = dag_hashes[
                searchsorted_inrange(dag_hashes, repo_pr_merge_hashes)
            ] == repo_pr_merge_hashes
            alive_indexes.extend(repo_order[begin_pos:end_pos][found])
        if (resolved := resolved.take(alive_indexes)).empty:
            return prs
        # take the commit that was committed the latest; if there are multiple, prefer the one
        # with pushed_date = null
        resolved.sort_values([PushCommit.committed_date.name, PushCommit.pushed_date.name],
                             ascending=False, inplace=True, na_position="first")
        resolved.drop_duplicates("pr_node_id", inplace=True)
        # patch the commit IDs and the hashes
        alive_node_ids = resolved["pr_node_id"].values
        if columns is PullRequest or PullRequest.merge_commit_id in columns:
            prs.loc[alive_node_ids, PullRequest.merge_commit_id.name] = \
                resolved["commit_node_id"].values
        if columns is PullRequest or PullRequest.merge_commit_sha in columns:
            prs.loc[alive_node_ids, PullRequest.merge_commit_sha.name] = resolved["sha"].values
        prs.loc[alive_node_ids, "dead"] = False
        return prs

    @classmethod
    async def _fetch_branch_dags(cls,
                                 repositories: Iterable[str],
                                 dags: Optional[Dict[str, DAG]],
                                 branches: pd.DataFrame,
                                 account: int,
                                 meta_ids: Tuple[int, ...],
                                 mdb: Database,
                                 pdb: Database,
                                 cache: Optional[aiomcache.Client],
                                 ) -> Dict[str, DAG]:
        if dags is None:
            dags = await fetch_precomputed_commit_history_dags(
                repositories, account, pdb, cache)
        return await fetch_repository_commits_no_branch_dates(
            dags, branches, BRANCH_FETCH_COMMITS_COLUMNS, True, account, meta_ids,
            mdb, pdb, cache)

    @classmethod
    @sentry_span
    async def _fetch_prs_by_filters(cls,
                                    time_from: Optional[datetime],
                                    time_to: datetime,
                                    repositories: Set[str],
                                    participants: PRParticipants,
                                    labels: LabelFilter,
                                    jira: JIRAFilter,
                                    exclude_inactive: bool,
                                    pr_blacklist: Optional[BinaryExpression],
                                    pr_whitelist: Optional[BinaryExpression],
                                    meta_ids: Tuple[int, ...],
                                    mdb: Database,
                                    cache: Optional[aiomcache.Client],
                                    columns=PullRequest,
                                    updated_min: Optional[datetime] = None,
                                    updated_max: Optional[datetime] = None,
                                    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        assert (updated_min is None) == (updated_max is None)
        filters = [
            (sql.case(
                [(PullRequest.closed, PullRequest.closed_at)],
                else_=sql.text("'3000-01-01'"),  # backed up with a DB index
            ) >= time_from) if time_from is not None else sql.true(),
            PullRequest.created_at < time_to,
            PullRequest.acc_id.in_(meta_ids),
            PullRequest.hidden.is_(False),
            PullRequest.repository_full_name.in_(repositories),
        ]
        if exclude_inactive and updated_min is None:
            # this does not provide 100% guarantee because it can be after time_to,
            # we need to properly filter later
            filters.append(PullRequest.updated_at >= time_from)
        if updated_min is not None:
            filters.append(PullRequest.updated_at.between(updated_min, updated_max))
        if pr_blacklist is not None:
            filters.append(pr_blacklist)
        if pr_whitelist is not None:
            filters.append(pr_whitelist)
        if len(participants) == 1:
            if PRParticipationKind.AUTHOR in participants:
                filters.append(PullRequest.user_login.in_(
                    participants[PRParticipationKind.AUTHOR]))
            elif PRParticipationKind.MERGER in participants:
                filters.append(
                    PullRequest.merged_by_login.in_(participants[PRParticipationKind.MERGER]))
        elif len(participants) == 2 and PRParticipationKind.AUTHOR in participants and \
                PRParticipationKind.MERGER in participants:
            filters.append(sql.or_(
                PullRequest.user_login.in_(participants[PRParticipationKind.AUTHOR]),
                PullRequest.merged_by_login.in_(participants[PRParticipationKind.MERGER]),
            ))
        if columns is PullRequest:
            selected_columns = [PullRequest]
            remove_acc_id = False
        else:
            selected_columns = columns = list(columns)
            if remove_acc_id := (PullRequest.acc_id not in selected_columns):
                selected_columns.append(PullRequest.acc_id)
            if PullRequest.merge_commit_id in columns or PullRequest.merge_commit_sha in columns:
                # needed to resolve rebased merge commits
                if PullRequest.number not in selected_columns:
                    selected_columns.append(PullRequest.number)
        if labels:
            singles, multiples = LabelFilter.split(labels.include)
            embedded_labels_query = not multiples
            if all_in_labels := (set(singles + list(chain.from_iterable(multiples)))):
                filters.append(
                    sql.exists().where(sql.and_(
                        PullRequestLabel.acc_id == PullRequest.acc_id,
                        PullRequestLabel.pull_request_node_id == PullRequest.node_id,
                        sql.func.lower(PullRequestLabel.name).in_(all_in_labels),
                    )))
            if labels.exclude:
                filters.append(
                    sql.not_(sql.exists().where(sql.and_(
                        PullRequestLabel.acc_id == PullRequest.acc_id,
                        PullRequestLabel.pull_request_node_id == PullRequest.node_id,
                        sql.func.lower(PullRequestLabel.name).in_(labels.exclude),
                    ))))
        if not jira:
            query = sql.select(selected_columns).where(sql.and_(*filters))
        else:
            query = await generate_jira_prs_query(
                filters, jira, None, mdb, cache, columns=selected_columns)
        prs = await read_sql_query(query, mdb, columns, index=PullRequest.node_id.name)
        if remove_acc_id:
            del prs[PullRequest.acc_id.name]
        if PullRequest.closed.name in prs:
            cls.adjust_pr_closed_merged_timestamps(prs)
        _, first_encounters = np.unique(prs.index.values, return_index=True)
        if len(first_encounters) < len(prs):
            prs = prs.take(first_encounters)
        if not labels or embedded_labels_query:
            return prs, None
        df_labels = await fetch_labels_to_filter(prs.index, meta_ids, mdb)
        left = cls.find_left_by_labels(
            prs.index, df_labels.index, df_labels[PullRequestLabel.name.name].values, labels)
        prs = prs.take(np.flatnonzero(prs.index.isin(left)))
        return prs, df_labels

    @staticmethod
    def adjust_pr_closed_merged_timestamps(prs_df: pd.DataFrame) -> None:
        """Force set `closed_at` and `merged_at` to NULL if not `closed`. Remove `closed`."""
        not_closed = ~prs_df[PullRequest.closed.name].values
        prs_df.loc[not_closed, PullRequest.closed_at.name] = pd.NaT
        prs_df.loc[not_closed, PullRequest.merged_at.name] = pd.NaT
        prs_df.drop(columns=PullRequest.closed.name, inplace=True)

    @classmethod
    @sentry_span
    async def _fetch_inactive_merged_unreleased_prs(
            cls,
            time_from: datetime,
            time_to: datetime,
            repos: Union[Set[str], KeysView[str]],
            participants: PRParticipants,
            labels: LabelFilter,
            jira: JIRAFilter,
            default_branches: Dict[str, str],
            release_settings: ReleaseSettings,
            has_logical_repos: bool,
            prefixer: Prefixer,
            account: int,
            meta_ids: Tuple[int, ...],
            mdb: Database,
            pdb: Database,
            cache: Optional[aiomcache.Client]) -> pd.DataFrame:
        node_id_map = await discover_inactive_merged_unreleased_prs(
            time_from, time_to, repos, participants, labels, default_branches, release_settings,
            prefixer, account, pdb, cache)
        if not jira:
            return await read_sql_query(sql.select([PullRequest])
                                        .where(PullRequest.node_id.in_(node_id_map)),
                                        mdb, PullRequest, index=PullRequest.node_id.name)
        df = await cls.filter_jira(node_id_map, jira, meta_ids, mdb, cache)
        if not has_logical_repos:
            return df
        append = defaultdict(list)
        node_ids = df.index.values
        repository_full_names = df[PullRequest.repository_full_name.name].values
        for i, (pr_node_id, physical_repo) in enumerate(zip(node_ids, repository_full_names)):
            logical_repos = node_id_map[pr_node_id]
            if physical_repo != (first_logical_repo := logical_repos[0]):
                repository_full_names[i] = first_logical_repo
            for logical_repo in logical_repos[1:]:
                append[logical_repo].append(i)
        if append:
            chunks = []
            for logical_repo, indexes in append.items():
                subdf = df.take(indexes)
                subdf[PullRequest.repository_full_name.name] = logical_repo
                chunks.append(subdf)
            df = pd.concat([df] + chunks)
        return df

    @classmethod
    @sentry_span
    async def filter_jira(cls,
                          pr_node_ids: Collection[int],
                          jira: JIRAFilter,
                          meta_ids: Tuple[int, ...],
                          mdb: Database,
                          cache: Optional[aiomcache.Client],
                          columns=PullRequest) -> pd.DataFrame:
        """Filter PRs by JIRA properties."""
        assert jira
        filters = [PullRequest.node_id.in_(pr_node_ids)]
        query = await generate_jira_prs_query(filters, jira, meta_ids, mdb, cache, columns=columns)
        # pr JOIN repo is always len(pr_node_ids)
        # speculate that m JOIN pr is ~0.5 of len(pr_node_ids), that is, half of PRs mapped to JIRA
        query = query \
            .with_statement_hint(f"Rows(pr repo #{len(pr_node_ids)})") \
            .with_statement_hint(f"Rows(m pr #{len(pr_node_ids) // 2})")
        return await read_sql_query(
            query, mdb, columns, index=PullRequest.node_id.name)

    @classmethod
    @sentry_span
    async def fetch_pr_deployments(cls,
                                   pr_node_ids: Iterable[int],
                                   account: int,
                                   pdb: Database,
                                   rdb: Database,
                                   ) -> pd.DataFrame:
        """Load the deployments for each PR node ID."""
        ghprd = GitHubPullRequestDeployment
        cols = [ghprd.pull_request_id, ghprd.deployment_name, ghprd.repository_full_name]
        df = await read_sql_query(
            sql.select(cols)
            .where(sql.and_(ghprd.acc_id == account,
                            ghprd.pull_request_id.in_any_values(pr_node_ids))),
            con=pdb, columns=cols, index=ghprd.deployment_name.name)
        cols = [DeploymentNotification.name,
                DeploymentNotification.environment,
                DeploymentNotification.conclusion,
                DeploymentNotification.finished_at]
        details = await read_sql_query(
            sql.select(cols)
            .where(sql.and_(DeploymentNotification.account_id == account,
                            DeploymentNotification.name.in_(df.index.values))),
            con=rdb, columns=cols, index=DeploymentNotification.name.name,
        )
        details.index.name = ghprd.deployment_name.name
        df = df.join(details)
        df.reset_index(inplace=True)
        df.set_index([
            ghprd.pull_request_id.name,
            ghprd.repository_full_name.name,
            ghprd.deployment_name.name,
        ], inplace=True)
        return df

    @staticmethod
    def _check_participants_compatibility(cached_participants: PRParticipants,
                                          participants: PRParticipants) -> bool:
        if not cached_participants:
            return True
        if not participants:
            return False
        for k, v in participants.items():
            if v - cached_participants.get(k, set()):
                return False
        return True

    @classmethod
    @sentry_span
    def _remove_spurious_prs(cls, time_from: datetime, dfs: PRDataFrames) -> None:
        old_releases = np.where(dfs.releases[Release.published_at.name] < time_from)[0]
        if len(old_releases) == 0:
            return
        cls._drop(dfs, dfs.releases.index[old_releases])

    @classmethod
    def _drop(cls, dfs: PRDataFrames, pr_ids: Collection[int]) -> None:
        if len(pr_ids) == 0:
            return
        for df in dfs.values():
            df.drop(pr_ids,
                    level=0 if isinstance(df.index, pd.MultiIndex) else None,
                    inplace=True,
                    errors="ignore")

    @classmethod
    @sentry_span
    def _find_drop_by_participants(cls,
                                   dfs: PRDataFrames,
                                   participants: PRParticipants,
                                   time_to: Optional[datetime],
                                   ) -> pd.Index:
        if not participants:
            return pd.Index([])
        if time_to is not None:
            for df_name, col in (("commits", PullRequestCommit.committed_date),
                                 ("reviews", PullRequestReview.created_at),
                                 ("review_comments", PullRequestReviewComment.created_at),
                                 ("review_requests", PullRequestReviewRequest.created_at),
                                 ("comments", PullRequestComment.created_at)):
                df = getattr(dfs, df_name)
                setattr(dfs, df_name, df.take(np.where(df[col.name] < time_to)[0]))
        passed = []
        dict_iter = (
            (dfs.prs, PullRequest.user_login, None, PRParticipationKind.AUTHOR),
            (dfs.prs, PullRequest.merged_by_login, PullRequest.merged_at, PRParticipationKind.MERGER),  # noqa
            (dfs.releases, Release.author, Release.published_at, PRParticipationKind.RELEASER),
        )
        for df, part_col, date_col, pk in dict_iter:
            col_parts = participants.get(pk)
            if not col_parts:
                continue
            mask = df[part_col.name].isin(col_parts)
            if time_to is not None and date_col is not None:
                mask &= df[date_col.name] < time_to
            passed.append(df.index.get_level_values(0).take(np.flatnonzero(mask)))
        reviewers = participants.get(PRParticipationKind.REVIEWER)
        if reviewers:
            ulkr = PullRequestReview.user_login.name
            ulkp = PullRequest.user_login.name
            user_logins = pd.merge(dfs.reviews[ulkr].droplevel(1), dfs.prs[ulkp],
                                   left_index=True, right_index=True, how="left", copy=False)
            ulkr += "_x"
            ulkp += "_y"
            passed.append(user_logins.index.take(np.where(
                (user_logins[ulkr] != user_logins[ulkp]) & user_logins[ulkr].isin(reviewers),
            )[0]).unique())
        for df, col, pk in (
                (dfs.comments, PullRequestComment.user_login, PRParticipationKind.COMMENTER),
                (dfs.commits, PullRequestCommit.author_login, PRParticipationKind.COMMIT_AUTHOR),
                (dfs.commits, PullRequestCommit.committer_login, PRParticipationKind.COMMIT_COMMITTER)):  # noqa
            col_parts = participants.get(pk)
            if not col_parts:
                continue
            passed.append(df.index.get_level_values(0).take(np.flatnonzero(
                df[col.name].isin(col_parts))).unique())
        while len(passed) > 1:
            new_passed = []
            for i in range(0, len(passed), 2):
                if i + 1 < len(passed):
                    new_passed.append(passed[i].union(passed[i + 1]))
                else:
                    new_passed.append(passed[i])
            passed = new_passed
        return dfs.prs.index.get_level_values(0).difference(passed[0])

    @classmethod
    @sentry_span
    def _find_drop_by_labels(cls, dfs: PRDataFrames, labels: LabelFilter) -> pd.Index:
        if not labels:
            return pd.Index([])
        df_labels_index = dfs.labels.index.get_level_values(0)
        df_labels_names = dfs.labels[PullRequestLabel.name.name].values
        pr_node_ids = dfs.prs.index.get_level_values(0)
        left = cls.find_left_by_labels(pr_node_ids, df_labels_index, df_labels_names, labels)
        if not labels.include:
            return df_labels_index.difference(left)
        return pr_node_ids.difference(left)

    @classmethod
    def find_left_by_labels(cls,
                            full_index: pd.Index,
                            df_labels_index: pd.Index,
                            df_labels_names: Sequence[str],
                            labels: LabelFilter) -> pd.Index:
        """
        Post-filter PRs by their loaded labels.

        :param full_index: All the PR node IDs, not just those that correspond to labeled PRs.
        :param df_labels_index: (PR node ID, label name) DataFrame index. There may be several \
                                rows for the same PR node ID.
        :param df_labels_names: (PR node ID, label name) DataFrame column.
        """
        left_include = left_exclude = None
        if labels.include:
            singles, multiples = LabelFilter.split(labels.include)
            left_include = df_labels_index.take(
                np.nonzero(np.in1d(df_labels_names, singles))[0],
            ).unique()
            for group in multiples:
                passed = df_labels_index
                for label in group:
                    passed = passed.intersection(
                        df_labels_index.take(np.nonzero(df_labels_names == label)[0]))
                    if passed.empty:
                        break
                left_include = left_include.union(passed)
        if labels.exclude:
            left_exclude = full_index.difference(df_labels_index.take(
                np.nonzero(np.in1d(df_labels_names, list(labels.exclude)))[0],
            ).unique())
        if labels.include:
            if labels.exclude:
                left = left_include.intersection(left_exclude)
            else:
                left = left_include
        else:
            left = left_exclude
        return left

    @classmethod
    @sentry_span
    def _find_drop_by_jira(cls, dfs: PRDataFrames, jira: JIRAFilter) -> pd.Index:
        if not jira:
            return pd.Index([])
        left = []
        jira_index = dfs.jiras.index.get_level_values(0)
        pr_node_ids = dfs.prs.index.get_level_values(0)
        if jira.labels:
            df_labels_names = dfs.jiras[Issue.labels.name].values
            df_labels_index = pd.Index(np.repeat(jira_index, [len(v) for v in df_labels_names]))
            df_labels_names = list(pd.core.common.flatten(df_labels_names))
            # if jira.labels.include is empty, we effectively drop all unmapped PRs
            # that is the desired behavior
            left.append(cls.find_left_by_labels(
                pr_node_ids, df_labels_index, df_labels_names, jira.labels))
        if jira.epics:
            left.append(jira_index.take(np.flatnonzero(
                dfs.jiras["epic"].isin(jira.epics).values)).unique())
        if jira.issue_types:
            left.append(dfs.jiras.index.get_level_values(0).take(np.flatnonzero(
                dfs.jiras[Issue.type.name].str.lower().isin(jira.issue_types).values)).unique())
        if jira.custom_projects:
            left.append(jira_index.take(np.flatnonzero(
                dfs.jiras[Issue.project_id.name].isin(jira.projects).values)).unique())
        result = left[0]
        for other in left[1:]:
            result = result.intersection(other)
        return pr_node_ids.difference(result)

    @classmethod
    @sentry_span
    def _find_drop_by_inactive(cls,
                               dfs: PRDataFrames,
                               time_from: datetime,
                               time_to: datetime) -> pd.Index:
        activities = [
            dfs.prs[PullRequest.created_at.name],
            dfs.prs[PullRequest.closed_at.name],
            dfs.commits[PullRequestCommit.committed_date.name],
            dfs.review_requests[PullRequestReviewRequest.created_at.name],
            dfs.reviews[PullRequestReview.created_at.name],
            dfs.comments[PullRequestComment.created_at.name],
            dfs.releases[Release.published_at.name],
            dfs.deployments[DeploymentNotification.finished_at.name],
        ]
        for df in activities:
            if df.index.nlevels > 1:
                df.index = df.index.get_level_values(0)
            df.name = "timestamp"
        activities = pd.concat(activities, copy=False)
        active_prs = activities.index.take(np.flatnonzero(
            activities.between(time_from, time_to).values)).drop_duplicates()
        inactive_prs = dfs.prs.index.get_level_values(0).difference(active_prs)
        return inactive_prs

    @staticmethod
    async def _read_filtered_models(model_cls: Base,
                                    node_ids: Collection[int],
                                    time_to: datetime,
                                    meta_ids: Tuple[int, ...],
                                    mdb: DatabaseLike,
                                    columns: Optional[List[InstrumentedAttribute]] = None,
                                    created_at=True,
                                    ) -> pd.DataFrame:
        if columns is not None:
            columns = [model_cls.pull_request_node_id, model_cls.node_id] + columns
        filters = [model_cls.pull_request_node_id.in_(node_ids),
                   model_cls.acc_id.in_(meta_ids)]
        if created_at:
            filters.append(model_cls.created_at < time_to)
        df = await read_sql_query(
            sql.select(columns or [model_cls]).where(sql.and_(*filters)),
            con=mdb,
            columns=columns or model_cls,
            index=[model_cls.pull_request_node_id.name, model_cls.node_id.name])
        return df

    @classmethod
    @sentry_span
    def _truncate_prs(cls,
                      dfs: PRDataFrames,
                      time_from: datetime,
                      time_to: datetime,
                      ) -> None:
        """
        Remove PRs outside of the given time range.

        This is used to correctly handle timezone offsets.
        """
        pr_node_ids_index = dfs.prs.index.get_level_values(0)

        # filter out PRs which were released before `time_from`
        unreleased = dfs.releases.index.get_level_values(0).take(np.flatnonzero(
            (dfs.releases[Release.published_at.name] < time_from).values))
        # closed but not merged in `[date_from, time_from]`
        unrejected = pr_node_ids_index.take(np.flatnonzero(
            ((dfs.prs[PullRequest.closed_at.name] < time_from) &
             dfs.prs[PullRequest.merged_at.name].isnull()).values))
        # created in `[time_to, date_to]`
        uncreated = pr_node_ids_index.take(np.flatnonzero(
            (dfs.prs[PullRequest.created_at.name] >= time_to).values))
        deployed = dfs.deployments.index.get_level_values(0).take(np.flatnonzero(
            dfs.deployments[DeploymentNotification.finished_at.name].between(time_from, time_to)
            .values))
        to_remove = unreleased \
            .union(unrejected) \
            .difference(deployed) \
            .union(uncreated)
        cls._drop(dfs, to_remove)

    @classmethod
    def _extract_missed_prs(cls,
                            ambiguous: Optional[Dict[str, Collection[int]]],
                            pr_blacklist: Optional[Collection[int]],
                            deployed_prs: pd.DataFrame,
                            matched_bys: Dict[str, ReleaseMatch],
                            ) -> Dict[str, np.ndarray]:
        missed_prs = {}
        if ambiguous is not None and pr_blacklist is not None:
            if isinstance(pr_blacklist, (dict, set)):
                pr_blacklist = list(pr_blacklist)
            if deployed_prs.empty:
                candidate_missing_node_ids = np.asarray(pr_blacklist)
            else:
                candidate_missing_node_ids = np.setdiff1d(
                    pr_blacklist, deployed_prs.index.values, assume_unique=True)
            if len(candidate_missing_node_ids):
                for repo, pr_node_ids in ambiguous.items():
                    if matched_bys[repo] == ReleaseMatch.tag:
                        repo_missed_node_ids = np.intersect1d(
                            pr_node_ids, candidate_missing_node_ids, assume_unique=True)
                        if len(repo_missed_node_ids):
                            missed_prs[repo] = repo_missed_node_ids
        return missed_prs

    @classmethod
    @sentry_span
    async def map_deployments_to_prs(cls,
                                     repositories: Union[Set[str], KeysView[str]],
                                     time_from: datetime,
                                     time_to: datetime,
                                     participants: PRParticipants,
                                     labels: LabelFilter,
                                     jira: JIRAFilter,
                                     updated_min: Optional[datetime],
                                     updated_max: Optional[datetime],
                                     branches: pd.DataFrame,
                                     dags: Optional[Dict[str, DAG]],
                                     account: int,
                                     meta_ids: Tuple[int, ...],
                                     mdb: Database,
                                     pdb: Database,
                                     cache: Optional[aiomcache.Client],
                                     pr_blacklist: Optional[List[int]] = None,
                                     fetch_branch_dags_task: Optional[asyncio.Task] = None,
                                     ) -> pd.DataFrame:
        """Load PRs which were deployed between `time_from` and `time_to` and merged before \
        `time_from`."""
        assert (updated_min is None) == (updated_max is None)
        ghprd = GitHubPullRequestDeployment
        precursor_prs = await pdb.fetch_all(
            sql.select([sql.distinct(ghprd.pull_request_id)])
            .where(sql.and_(ghprd.acc_id == account,
                            ghprd.repository_full_name.in_(repositories),
                            ghprd.finished_at.between(time_from, time_to))))
        precursor_prs = {r[0] for r in precursor_prs}
        if pr_blacklist:
            if isinstance(pr_blacklist, dict):
                precursor_prs -= pr_blacklist.keys()
            elif isinstance(pr_blacklist, set):
                precursor_prs -= pr_blacklist
            else:
                precursor_prs -= set(pr_blacklist)
        if not precursor_prs:
            return pd.DataFrame()
        prs, *_ = await cls.fetch_prs(
            None, time_to, repositories, participants, labels, jira,
            False, None, PullRequest.node_id.in_(precursor_prs), branches, dags, account, meta_ids,
            mdb, pdb, cache, updated_min=updated_min, updated_max=updated_max,
            fetch_branch_dags_task=fetch_branch_dags_task)
        return prs

    @staticmethod
    def _iter_by_split(arr: np.ndarray, splits: np.ndarray) -> Generator[np.ndarray, None, None]:
        for beg, end in zip(splits[:-1], splits[1:]):
            yield arr[beg:end]


class ReviewResolution(Enum):
    """Possible review "state"-s in the metadata DB."""

    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"


class ImpossiblePullRequest(Exception):
    """Raised by PullRequestFactsMiner._compile() on broken PRs."""


class PullRequestFactsMiner:
    """Extract the pull request event timestamps from MinedPullRequest-s."""

    log = logging.getLogger("%s.PullRequestFactsMiner" % metadata.__package__)
    dummy_reviews = pd.Series(["INVALID", pd.NaT],
                              index=[PullRequestReview.state.name,
                                     PullRequestReview.submitted_at.name])

    def __init__(self, bots: Set[str]):
        """Require the set of bots to be preloaded."""
        self._bots = np.sort(list(bots)).astype("S")

    def __call__(self, pr: MinedPullRequest) -> PullRequestFacts:
        """
        Extract the pull request event timestamps from a MinedPullRequest.

        May raise ImpossiblePullRequest if the PR has an "impossible" state like
        created after closed.
        """
        created = pr.pr[PullRequest.created_at.name]
        if created != created:
            raise ImpossiblePullRequest()
        merged = pr.pr[PullRequest.merged_at.name]
        if merged != merged:
            merged = None
        closed = pr.pr[PullRequest.closed_at.name]
        if closed != closed:
            closed = None
        if merged and not closed:
            self.log.error("[DEV-508] PR %s (%s#%d) is merged at %s but not closed",
                           pr.pr[PullRequest.node_id.name],
                           pr.pr[PullRequest.repository_full_name.name],
                           pr.pr[PullRequest.number.name],
                           merged)
            closed = merged
        # we don't need these indexes
        pr.comments.reset_index(inplace=True, drop=True)
        pr.reviews.reset_index(inplace=True, drop=True)
        # first_commit usually points at min(authored_date)
        # last_commit usually points at max(committed_date)
        # but DEV-2734 has taught that authored_date may be greater than committed_date
        first_commit = nonemin(
            pr.commits[PullRequestCommit.authored_date.name].nonemin(),
            pr.commits[PullRequestCommit.committed_date.name].nonemin(),
        )
        last_commit = nonemax(
            pr.commits[PullRequestCommit.committed_date.name].nonemax(),
            pr.commits[PullRequestCommit.authored_date.name].nonemax(),
        )
        # convert to "S" dtype to enable sorting in np.in1d
        authored_comments = \
            pr.comments[PullRequestReviewComment.user_login.name].values.astype("S")
        if (author_login := pr.pr[PullRequest.user_login.name]) is not None:
            author_login = author_login.encode()
        external_comments_times = pr.comments[PullRequestComment.created_at.name].take(
            np.flatnonzero((authored_comments != author_login) &
                           np.in1d(authored_comments, self._bots, invert=True)))
        first_comment = nonemin(
            pr.review_comments[PullRequestReviewComment.created_at.name].nonemin(),
            pr.reviews[PullRequestReview.submitted_at.name].nonemin(),
            external_comments_times.nonemin())
        if closed and first_comment and first_comment > closed:
            first_comment = None
        first_comment_on_first_review = first_comment or merged
        if first_comment_on_first_review:
            committed_dates = pr.commits[PullRequestCommit.committed_date.name]
            last_commit_before_first_review = committed_dates.take(np.where(
                committed_dates <= first_comment_on_first_review)[0]).nonemax()
            if not (last_commit_before_first_review_own := bool(last_commit_before_first_review)):
                last_commit_before_first_review = first_comment_on_first_review
            # force pushes that were lost
            first_commit = nonemin(first_commit, last_commit_before_first_review)
            last_commit = nonemax(last_commit, first_commit)
            first_review_request_backup = nonemin(
                nonemax(created, last_commit_before_first_review),
                first_comment_on_first_review)
        else:
            last_commit_before_first_review = None
            last_commit_before_first_review_own = False
            first_review_request_backup = None
        first_review_request_exact = \
            pr.review_requests[PullRequestReviewRequest.created_at.name].nonemin()
        if first_review_request_exact and first_review_request_exact < created:
            # DEV-1610: there are lags possible
            first_review_request_exact = created
        first_review_request = first_review_request_exact
        if first_review_request_backup and first_review_request and \
                first_review_request > first_comment_on_first_review:
            # we cannot request a review after we received a review
            first_review_request = first_review_request_backup
        else:
            first_review_request = first_review_request or first_review_request_backup
        # ensure that the first review request is not earlier than the last commit before
        # the first review
        if last_commit_before_first_review_own and \
                last_commit_before_first_review > first_review_request:
            first_review_request = last_commit_before_first_review or first_review_request
        if closed and first_review_request and first_review_request > closed:
            first_review_request = None
        review_submitted_ats = pr.reviews[PullRequestReview.submitted_at.name]

        if closed:
            # it is possible to approve/reject after closing the PR
            # you start the review, then somebody closes the PR, then you submit the review
            try:
                last_review = pd.Timestamp(review_submitted_ats.values[
                    (review_submitted_ats.values <= closed.to_numpy())
                ].max(), tz=timezone.utc)
            except ValueError:
                last_review = None
            last_review = nonemax(
                last_review,
                external_comments_times.take(np.flatnonzero(
                    external_comments_times <= closed)).nonemax())
        else:
            last_review = review_submitted_ats.nonemax() or \
                nonemin(external_comments_times.nonemax())
        if not first_review_request:
            assert not last_review, pr.pr[PullRequest.node_id.name]
        if closed:
            min_merged_closed = nonemin(merged, closed).to_numpy()
            reviews_before_close = \
                pr.reviews[PullRequestReview.submitted_at.name].values <= min_merged_closed
            if reviews_before_close.all():
                reviews_before_close = pr.reviews
            else:
                reviews_before_close = pr.reviews.take(np.where(reviews_before_close)[0])
                reviews_before_close.reset_index(drop=True, inplace=True)
        else:
            reviews_before_close = pr.reviews
        # the most recent review for each reviewer
        if reviews_before_close.empty:
            # express lane
            grouped_reviews = self.dummy_reviews
        else:
            review_logins = \
                reviews_before_close[PullRequestReview.user_login.name].values.astype("S")
            human_review_mask = np.in1d(review_logins, self._bots, invert=True)
            if not human_review_mask.all():
                reviews_before_close = reviews_before_close.take(np.nonzero(human_review_mask)[0])
                reviews_before_close.reset_index(inplace=True, drop=True)
                review_logins = review_logins[human_review_mask]
            if len(np.unique(review_logins)) == 1:
                # fast lane
                grouped_reviews = reviews_before_close._ixs(
                    reviews_before_close[PullRequestReview.submitted_at.name].values.argmax())
            else:
                # the most recent review for each reviewer
                latest_review_ixs = [
                    ixs[0] for ixs in
                    reviews_before_close[[PullRequestReview.user_login.name,
                                          PullRequestReview.submitted_at.name]]
                    .take(np.where(reviews_before_close[PullRequestReview.state.name] !=
                                   ReviewResolution.COMMENTED.value)[0])
                    .sort_values([PullRequestReview.submitted_at.name], ascending=False)
                    .groupby(PullRequestReview.user_login.name, sort=False)
                    .grouper.groups.values()
                ]
                grouped_reviews = {
                    k: reviews_before_close[k].take(latest_review_ixs)
                    for k in (PullRequestReview.state.name, PullRequestReview.submitted_at.name)}
        grouped_reviews_states = grouped_reviews[PullRequestReview.state.name]
        if isinstance(grouped_reviews_states, str):
            changes_requested = grouped_reviews_states == ReviewResolution.CHANGES_REQUESTED.value
        else:
            changes_requested = (
                grouped_reviews_states.values == ReviewResolution.CHANGES_REQUESTED.value
            ).any()
        if changes_requested:
            # merged with negative reviews
            approved = None
        else:
            if isinstance(grouped_reviews_states, str):
                if grouped_reviews_states == ReviewResolution.APPROVED.value:
                    approved = grouped_reviews[PullRequestReview.submitted_at.name]
                else:
                    approved = None
            else:
                approved = grouped_reviews[PullRequestReview.submitted_at.name].take(
                    np.where(grouped_reviews_states == ReviewResolution.APPROVED.value)[0],
                ).nonemax()
        released = pr.release[Release.published_at.name]
        if released != released:
            released = None
        additions = pr.pr[PullRequest.additions.name]
        deletions = pr.pr[PullRequest.deletions.name]
        if additions is None or deletions is None:
            self.log.error("NULL in PR additions or deletions: %s (%s#%d): +%s -%s",
                           pr.pr[PullRequest.node_id.name],
                           pr.pr[PullRequest.repository_full_name.name],
                           pr.pr[PullRequest.number.name],
                           additions, deletions)
            raise ImpossiblePullRequest()
        size = additions + deletions
        force_push_dropped = pr.release[matched_by_column] == ReleaseMatch.force_push_drop
        done = bool(released or force_push_dropped or (closed and not merged))
        work_began = nonemin(created, first_commit)
        ts_dtype = "datetime64[ns]"
        reviews = np.sort(reviews_before_close[PullRequestReview.submitted_at.name].values) \
            .astype(ts_dtype)
        activity_days = np.concatenate([
            np.array([created.to_numpy() if created is not None else None,
                      closed.to_numpy() if closed is not None else None,
                      released.to_numpy() if released is not None else None], dtype=ts_dtype),
            pr.commits[PullRequestCommit.committed_date.name].values,
            pr.review_requests[PullRequestReviewRequest.created_at.name].values,
            pr.reviews[PullRequestReview.created_at.name].values,
            pr.comments[PullRequestComment.created_at.name].values,
        ]).astype(ts_dtype).astype("datetime64[D]")
        activity_days = \
            np.unique(activity_days[activity_days == activity_days]).astype(ts_dtype)
        review_comment_authors = \
            pr.review_comments[PullRequestReviewComment.user_login.name].values.astype("S")
        if len(review_comment_authors) > 0:
            human_review_comments = np.in1d(review_comment_authors, self._bots, invert=True).sum()
        else:
            human_review_comments = 0
        participants = len(np.setdiff1d(np.unique(np.concatenate([
            np.array([pr.pr[PullRequest.user_login.name], pr.pr[PullRequest.merged_by_login.name]],
                     dtype="S"),
            pr.commits[PullRequestCommit.author_login.name].values.astype("S"),
            pr.commits[PullRequestCommit.committer_login.name].values.astype("S"),
            review_comment_authors,
            pr.comments[PullRequestComment.user_login.name].values.astype("S"),
            pr.reviews[PullRequestReview.user_login.name].values.astype("S"),
        ])), self._bots, assume_unique=True))
        environments = \
            pr.deployments[DeploymentNotification.environment.name].values.astype("U", copy=False)
        deployment_conclusions = np.fromiter(
            (DeploymentConclusion[s]
             for s in pr.deployments[DeploymentNotification.conclusion.name].values),
            int, len(pr.deployments),
        )
        deployed = pr.deployments[DeploymentNotification.finished_at.name].values.astype(
            "datetime64[s]")
        facts = PullRequestFacts.from_fields(
            created=created,
            first_commit=first_commit,
            work_began=work_began,
            last_commit_before_first_review=last_commit_before_first_review,
            last_commit=last_commit,
            merged=merged,
            first_comment_on_first_review=first_comment_on_first_review,
            first_review_request=first_review_request,
            first_review_request_exact=first_review_request_exact,
            last_review=last_review,
            approved=approved,
            released=released,
            closed=closed,
            done=done,
            reviews=reviews,
            activity_days=activity_days,
            size=size,
            force_push_dropped=force_push_dropped,
            repository_full_name=pr.pr[PullRequest.repository_full_name.name],
            author=pr.pr[PullRequest.user_login.name],
            merger=pr.pr[PullRequest.merged_by_login.name],
            releaser=pr.release[Release.author.name],
            review_comments=human_review_comments,
            participants=participants,
            jira_ids=pr.jiras.index.values.tolist(),
            deployments=pr.deployments.index.get_level_values(1).values,
            environments=environments,
            deployment_conclusions=deployment_conclusions,
            deployed=deployed,
        )
        self._validate(facts, pr.pr[PullRequest.htmlurl.name])
        return facts

    def _validate(self, facts: PullRequestFacts, url: str) -> None:
        """Run sanity checks to ensure consistency."""
        assert facts.created == facts.created
        if not facts.closed:
            return
        if facts.last_commit and facts.last_commit > facts.closed:
            self.log.error("%s is impossible: closed %s but last commit %s: delta %s",
                           url, facts.closed, facts.last_commit,
                           facts.closed - facts.last_commit)
            raise ImpossiblePullRequest()
        if facts.created > facts.closed:
            self.log.error("%s is impossible: closed %s but created %s: delta %s",
                           url, facts.closed, facts.created,
                           facts.closed - facts.created)
            raise ImpossiblePullRequest()
        if facts.merged and facts.released and facts.merged > facts.released:
            self.log.error("%s is impossible: merged %s but released %s: delta %s",
                           url, facts.merged, facts.released,
                           facts.released - facts.merged)
            raise ImpossiblePullRequest()
