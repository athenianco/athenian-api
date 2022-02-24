import asyncio
from datetime import datetime, timedelta, timezone
import logging
import pickle
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Union

import aiomcache
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, func, join, or_, select, union_all
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api import metadata
from athenian.api.async_utils import gather, postprocess_datetime, read_sql_query
from athenian.api.cache import cached, max_exptime, middle_term_exptime
from athenian.api.controllers.logical_repos import coerce_logical_repos, contains_logical_repos, \
    drop_logical_repo
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.branches import load_branch_commit_dates
from athenian.api.controllers.miners.github.commit import DAG, \
    fetch_precomputed_commit_history_dags, fetch_repository_commits, RELEASE_FETCH_COMMITS_COLUMNS
from athenian.api.controllers.miners.github.dag_accelerated import extract_subdag, \
    mark_dag_access, mark_dag_parents, searchsorted_inrange
from athenian.api.controllers.miners.github.label import fetch_labels_to_filter
from athenian.api.controllers.miners.github.logical import split_logical_prs
from athenian.api.controllers.miners.github.precomputed_prs import \
    DonePRFactsLoader, MergedPRFactsLoader, update_unreleased_prs
from athenian.api.controllers.miners.github.release_load import dummy_releases_df, \
    group_repos_by_release_match, match_groups_to_sql, ReleaseLoader
from athenian.api.controllers.miners.github.released_pr import index_name, new_released_prs_df, \
    release_columns
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.controllers.miners.types import nonemax, PullRequestFactsMap
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.settings import LogicalPRSettings, LogicalRepositorySettings, \
    ReleaseMatch, ReleaseSettings
from athenian.api.db import add_pdb_hits, add_pdb_misses, Database, insert_or_ignore
from athenian.api.defer import defer
from athenian.api.models.metadata.github import NodeCommit, NodeRepository, PullRequest, \
    PullRequestLabel, PushCommit, Release
from athenian.api.models.precomputed.models import GitHubRelease as PrecomputedRelease, \
    GitHubRepository
from athenian.api.tracing import sentry_span


async def load_commit_dags(releases: pd.DataFrame,
                           account: int,
                           meta_ids: Tuple[int, ...],
                           mdb: Database,
                           pdb: Database,
                           cache: Optional[aiomcache.Client],
                           ) -> Dict[str, DAG]:
    """Produce the commit history DAGs which should contain the specified releases."""
    pdags = await fetch_precomputed_commit_history_dags(
        releases[Release.repository_full_name.name].unique(), account, pdb, cache)
    return await fetch_repository_commits(
        pdags, releases, RELEASE_FETCH_COMMITS_COLUMNS, False, account, meta_ids, mdb, pdb, cache)


class PullRequestToReleaseMapper:
    """Mapper from pull requests to releases."""

    @classmethod
    @sentry_span
    async def map_prs_to_releases(cls,
                                  prs: pd.DataFrame,
                                  releases: pd.DataFrame,
                                  matched_bys: Dict[str, ReleaseMatch],
                                  branches: pd.DataFrame,
                                  default_branches: Dict[str, str],
                                  time_to: datetime,
                                  dags: Dict[str, DAG],
                                  release_settings: ReleaseSettings,
                                  prefixer: Prefixer,
                                  account: int,
                                  meta_ids: Tuple[int, ...],
                                  mdb: Database,
                                  pdb: Database,
                                  cache: Optional[aiomcache.Client],
                                  labels: Optional[pd.DataFrame] = None,
                                  ) -> Tuple[pd.DataFrame,
                                             PullRequestFactsMap,
                                             asyncio.Event]:
        """
        Match the merged pull requests to the nearest releases that include them.

        :return: 1. pd.DataFrame with the mapped PRs. \
                 2. Precomputed facts about unreleased merged PRs. \
                 3. Synchronization for updating the pdb table with merged unreleased PRs.
        """
        assert isinstance(time_to, datetime)
        assert isinstance(mdb, Database)
        assert isinstance(pdb, Database)
        assert prs.index.nlevels == 2
        pr_releases = new_released_prs_df()
        unreleased_prs_event = asyncio.Event()
        if prs.empty:
            unreleased_prs_event.set()
            return pr_releases, {}, unreleased_prs_event
        _, unreleased_prs, precomputed_pr_releases = await gather(
            load_branch_commit_dates(branches, meta_ids, mdb),
            MergedPRFactsLoader.load_merged_unreleased_pull_request_facts(
                prs, nonemax(releases[Release.published_at.name].nonemax(), time_to),
                LabelFilter.empty(), matched_bys, default_branches, release_settings,
                prefixer, account, pdb),
            DonePRFactsLoader.load_precomputed_pr_releases(
                prs, time_to, matched_bys, default_branches, release_settings,
                prefixer, account, pdb, cache),
        )
        add_pdb_hits(pdb, "map_prs_to_releases/released", len(precomputed_pr_releases))
        add_pdb_hits(pdb, "map_prs_to_releases/unreleased", len(unreleased_prs))
        pr_releases = precomputed_pr_releases
        merged_prs = prs[~prs.index.isin(pr_releases.index.union(unreleased_prs.keys()))]
        if merged_prs.empty:
            unreleased_prs_event.set()
            return pr_releases, unreleased_prs, unreleased_prs_event

        labels, missed_released_prs, dead_prs = await gather(
            cls._fetch_labels(merged_prs.index.get_level_values(0).values, labels, meta_ids, mdb),
            cls._map_prs_to_releases(merged_prs, dags, releases),
            cls._find_dead_merged_prs(merged_prs),
        )
        assert missed_released_prs.index.nlevels == 2
        assert dead_prs.index.nlevels == 2
        # PRs may wrongly classify as dead although they are really released; remove the conflicts
        dead_prs.drop(index=missed_released_prs.index, inplace=True, errors="ignore")
        add_pdb_misses(pdb, "map_prs_to_releases/released", len(missed_released_prs))
        add_pdb_misses(pdb, "map_prs_to_releases/dead", len(dead_prs))
        add_pdb_misses(pdb, "map_prs_to_releases/unreleased",
                       len(merged_prs) - len(missed_released_prs) - len(dead_prs))
        if not dead_prs.empty:
            if not missed_released_prs.empty:
                missed_released_prs = pd.concat([missed_released_prs, dead_prs])
            else:
                missed_released_prs = dead_prs
        await defer(update_unreleased_prs(
            merged_prs, missed_released_prs, time_to, labels, matched_bys, default_branches,
            release_settings, account, pdb, unreleased_prs_event),
            "update_unreleased_prs(%d, %d)" % (len(merged_prs), len(missed_released_prs)))
        return pr_releases.append(missed_released_prs), unreleased_prs, unreleased_prs_event

    @classmethod
    async def _map_prs_to_releases(cls,
                                   prs: pd.DataFrame,
                                   dags: Dict[str, DAG],
                                   releases: pd.DataFrame,
                                   ) -> pd.DataFrame:
        if prs.empty:
            return new_released_prs_df()
        assert prs.index.nlevels == 2
        release_repos = releases[Release.repository_full_name.name].values
        unique_release_repos, release_index_map, release_repo_counts = np.unique(
            release_repos, return_inverse=True, return_counts=True)
        # stable sort to preserve the decreasing order by published_at
        release_repo_order = np.argsort(release_index_map, kind="stable")
        ordered_release_shas = releases[Release.sha.name].values[release_repo_order].astype("S40")
        release_repo_offsets = np.zeros(len(release_repo_counts) + 1, dtype=int)
        np.cumsum(release_repo_counts, out=release_repo_offsets[1:])
        pr_repos = prs.index.get_level_values(1).values
        unique_pr_repos, pr_index_map, pr_repo_counts = np.unique(
            pr_repos, return_inverse=True, return_counts=True)
        pr_repo_order = np.argsort(pr_index_map)
        pr_merge_hashes = \
            prs[PullRequest.merge_commit_sha.name].values[pr_repo_order].astype("S40")
        pr_merged_at = prs[PullRequest.merged_at.name].values[pr_repo_order].astype(
            releases[Release.published_at.name].values.dtype, copy=False)
        pr_node_ids = prs.index.get_level_values(0).values[pr_repo_order]
        pr_repo_offsets = np.zeros(len(pr_repo_counts) + 1, dtype=int)
        np.cumsum(pr_repo_counts, out=pr_repo_offsets[1:])
        release_pos = pr_pos = 0
        released_prs = []
        log = logging.getLogger("%s.map_prs_to_releases" % metadata.__package__)
        while release_pos < len(unique_release_repos) and pr_pos < len(unique_pr_repos):
            release_repo = unique_release_repos[release_pos]
            pr_repo = unique_pr_repos[pr_pos]
            if release_repo == pr_repo:
                hashes, vertexes, edges = dags[drop_logical_repo(pr_repo)]
                if len(hashes) == 0:
                    log.error("very suspicious: empty DAG for %s", pr_repo)
                release_beg = release_repo_offsets[release_pos]
                release_end = release_repo_offsets[release_pos + 1]
                ownership = mark_dag_access(
                    hashes, vertexes, edges, ordered_release_shas[release_beg:release_end], True)
                unmatched = np.flatnonzero(ownership == (release_end - release_beg))
                if len(unmatched) > 0:
                    hashes = np.delete(hashes, unmatched)
                    ownership = np.delete(ownership, unmatched)
                if len(hashes) == 0:
                    release_pos += 1
                    continue
                pr_beg = pr_repo_offsets[pr_pos]
                pr_end = pr_repo_offsets[pr_pos + 1]
                merge_hashes = pr_merge_hashes[pr_beg:pr_end]
                merges_found = searchsorted_inrange(hashes, merge_hashes)
                found_mask = hashes[merges_found] == merge_hashes
                found_releases = releases[release_columns].take(
                    release_repo_order[release_beg:release_end]
                    [ownership[merges_found[found_mask]]])
                if not found_releases.empty:
                    found_releases[Release.published_at.name] = np.maximum(
                        found_releases[Release.published_at.name].values,
                        pr_merged_at[pr_beg:pr_end][found_mask])
                    found_releases[index_name] = pr_node_ids[pr_beg:pr_end][found_mask]
                    released_prs.append(found_releases)
                release_pos += 1
                pr_pos += 1
            elif release_repo < pr_repo:
                release_pos += 1
            else:
                pr_pos += 1
        if released_prs:
            released_prs = pd.concat(released_prs, copy=False)
            released_prs.set_index([index_name, Release.repository_full_name.name], inplace=True)
        else:
            released_prs = new_released_prs_df()
        return postprocess_datetime(released_prs)

    @classmethod
    @sentry_span
    async def _find_dead_merged_prs(cls, prs: pd.DataFrame) -> pd.DataFrame:
        assert prs.index.nlevels == 2
        dead_mask = prs["dead"].values.astype(bool, copy=False)
        node_ids = prs.index.get_level_values(0).values
        repos = prs.index.get_level_values(1).values
        dead_prs = [
            (pr_id, None, None, None, None, None, repo, ReleaseMatch.force_push_drop)
            for repo, pr_id in zip(repos[dead_mask], node_ids[dead_mask])
        ]
        return new_released_prs_df(dead_prs)

    @classmethod
    @sentry_span
    async def _fetch_labels(cls,
                            node_ids: Iterable[int],
                            df: Optional[pd.DataFrame],
                            meta_ids: Tuple[int, ...],
                            mdb: Database,
                            ) -> Dict[int, List[str]]:
        if df is not None:
            labels = {}
            for node_id, name in zip(df.index.get_level_values(0).values,
                                     df[PullRequestLabel.name.name].values):
                labels.setdefault(node_id, []).append(name)
            return labels
        rows = await mdb.fetch_all(
            select([PullRequestLabel.pull_request_node_id, func.lower(PullRequestLabel.name)])
            .where(and_(PullRequestLabel.pull_request_node_id.in_(node_ids),
                        PullRequestLabel.acc_id.in_(meta_ids))))
        labels = {}
        for row in rows:
            node_id, label = row[0], row[1]
            labels.setdefault(node_id, []).append(label)
        return labels


class ReleaseToPullRequestMapper:
    """Mapper from releases to pull requests."""

    release_loader = ReleaseLoader

    @classmethod
    @sentry_span
    async def map_releases_to_prs(cls,
                                  repos: Collection[str],
                                  branches: pd.DataFrame,
                                  default_branches: Dict[str, str],
                                  time_from: datetime,
                                  time_to: datetime,
                                  authors: Collection[str],
                                  mergers: Collection[str],
                                  jira: JIRAFilter,
                                  release_settings: ReleaseSettings,
                                  logical_settings: LogicalRepositorySettings,
                                  updated_min: Optional[datetime],
                                  updated_max: Optional[datetime],
                                  pdags: Optional[Dict[str, DAG]],
                                  prefixer: Prefixer,
                                  account: int,
                                  meta_ids: Tuple[int, ...],
                                  mdb: Database,
                                  pdb: Database,
                                  rdb: Database,
                                  cache: Optional[aiomcache.Client],
                                  pr_blacklist: Optional[BinaryExpression] = None,
                                  pr_whitelist: Optional[BinaryExpression] = None,
                                  truncate: bool = True,
                                  precomputed_observed: Optional[Tuple[
                                      np.ndarray, np.ndarray]] = None,
                                  ) -> Union[Tuple[pd.DataFrame,
                                                   pd.DataFrame,
                                                   ReleaseSettings,
                                                   Dict[str, ReleaseMatch],
                                                   Dict[str, DAG],
                                                   Tuple[np.ndarray, np.ndarray]],
                                             pd.DataFrame]:
        """Find pull requests which were released between `time_from` and `time_to` but merged before \
        `time_from`.

        The returned DataFrame-s with releases are already with logical repositories.

        :param authors: Required PR commit_authors.
        :param mergers: Required PR mergers.
        :param truncate: Do not load releases after `time_to`.
        :param precomputed_observed: Saved all_observed_commits and all_observed_repos from \
                                     the previous identical invocation. \
                                     See PullRequestMiner._mine().
        :return: pd.DataFrame with found PRs that were created before `time_from` and released \
                 between `time_from` and `time_to` \
                 (the rest exists if `precomputed_observed` is None) + \
                 pd.DataFrame with the discovered releases between \
                 `time_from` and `time_to` (today if not `truncate`) \
                 +\
                 holistic release settings that enforce the happened release matches in \
                 [`time_from`, `time_to`] \
                 + \
                 `matched_bys` so that we don't have to compute that mapping again. \
                 + \
                 commit DAGs that contain the relevant releases. \
                 + \
                 observed commits and repositories (precomputed cache for \
                 the second call if needed).
        """
        assert isinstance(time_from, datetime)
        assert isinstance(time_to, datetime)
        assert isinstance(mdb, Database)
        assert isinstance(pdb, Database)
        assert isinstance(pr_blacklist, (BinaryExpression, type(None)))
        assert isinstance(pr_whitelist, (BinaryExpression, type(None)))
        assert (updated_min is None) == (updated_max is None)

        if precomputed_observed is None:
            (
                all_observed_commits, all_observed_repos,
                releases_in_time_range, release_settings, matched_bys, dags,
            ) = await cls._map_releases_to_prs_observe(
                repos, branches, default_branches, time_from, time_to,
                release_settings, logical_settings, pdags,
                prefixer, account, meta_ids, mdb, pdb, rdb, cache, truncate)
        else:
            all_observed_commits, all_observed_repos = precomputed_observed

        if len(all_observed_commits):
            prs = await cls._find_old_released_prs(
                all_observed_commits, all_observed_repos, time_from, authors, mergers, jira,
                updated_min, updated_max, pr_blacklist, pr_whitelist, logical_settings,
                prefixer, meta_ids, mdb, cache)
        else:
            prs = pd.DataFrame(columns=[c.name for c in PullRequest.__table__.columns
                                        if c.name != PullRequest.node_id.name])
            prs.index = pd.Index([], name=PullRequest.node_id.name)
        prs["dead"] = False
        if precomputed_observed is None:
            return (
                prs, releases_in_time_range, release_settings, matched_bys, dags,
                (all_observed_commits, all_observed_repos),
            )
        return prs

    @classmethod
    @sentry_span
    async def _map_releases_to_prs_observe(cls,
                                           repos: Collection[str],  # logical
                                           branches: pd.DataFrame,
                                           default_branches: Dict[str, str],
                                           time_from: datetime,
                                           time_to: datetime,
                                           release_settings: ReleaseSettings,
                                           logical_settings: LogicalRepositorySettings,
                                           pdags: Optional[Dict[str, DAG]],
                                           prefixer: Prefixer,
                                           account: int,
                                           meta_ids: Tuple[int, ...],
                                           mdb: Database,
                                           pdb: Database,
                                           rdb: Database,
                                           cache: Optional[aiomcache.Client],
                                           truncate: bool,
                                           ) -> Tuple[np.ndarray,
                                                      np.ndarray,
                                                      pd.DataFrame,
                                                      ReleaseSettings,
                                                      Dict[str, ReleaseMatch],
                                                      Dict[str, DAG]]:
        matched_bys, releases, releases_in_time_range, release_settings, dags = \
            await cls._find_releases_for_matching_prs(
                repos, branches, default_branches, time_from, time_to, not truncate,
                release_settings, logical_settings, pdags, prefixer, account, meta_ids,
                mdb, pdb, rdb, cache)

        rpak = Release.published_at.name
        rrfnk = Release.repository_full_name.name
        all_observed_repos = []
        all_observed_commits = []
        # find the released commit hashes by two DAG traversals
        with sentry_sdk.start_span(op="all_observed_*"):
            for repo, repo_releases in releases.groupby(rrfnk, sort=False):
                if (repo_releases[rpak] >= time_from).any():
                    observed_commits = cls._extract_released_commits(
                        repo_releases, dags[drop_logical_repo(repo)], time_from)
                    if len(observed_commits):
                        all_observed_commits.append(observed_commits)
                        all_observed_repos.append(np.full(
                            len(observed_commits), repo, dtype=f"S{len(repo)}"))
        if all_observed_commits:
            all_observed_repos = np.concatenate(all_observed_repos)
            all_observed_commits = np.concatenate(all_observed_commits)
            order = np.argsort(all_observed_commits)
            all_observed_commits = all_observed_commits[order]
            all_observed_repos = all_observed_repos[order]
        else:
            all_observed_commits = all_observed_repos = np.array([])
        return (
            all_observed_commits, all_observed_repos,
            releases_in_time_range, release_settings, matched_bys, dags,
        )

    @classmethod
    @sentry_span
    async def _find_releases_for_matching_prs(
            cls,
            repos: Iterable[str],  # logical
            branches: pd.DataFrame,
            default_branches: Dict[str, str],
            time_from: datetime,
            time_to: datetime,
            until_today: bool,
            release_settings: ReleaseSettings,
            logical_settings: LogicalRepositorySettings,
            pdags: Optional[Dict[str, DAG]],
            prefixer: Prefixer,
            account: int,
            meta_ids: Tuple[int, ...],
            mdb: Database,
            pdb: Database,
            rdb: Database,
            cache: Optional[aiomcache.Client],
            releases_in_time_range: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict[str, ReleaseMatch],
               pd.DataFrame,
               pd.DataFrame,
               ReleaseSettings,
               Dict[str, DAG]]:
        """
        Load releases with sufficient history depth.

        1. Load releases between `time_from` and `time_to`, record the effective release matches.
        2. Use those matches to load enough releases before `time_from` to ensure we don't get \
           "release leakages" in the commit DAG.
        3. Optionally, use those matches to load all the releases after `time_to`.
        """
        if releases_in_time_range is None:
            # we have to load releases in two separate batches: before and after time_from
            # that's because the release strategy can change depending on the time range
            # see ENG-710 and ENG-725
            releases_in_time_range, matched_bys = await cls.release_loader.load_releases(
                repos, branches, default_branches, time_from, time_to,
                release_settings, logical_settings, prefixer, account, meta_ids,
                mdb, pdb, rdb, cache)
        else:
            matched_bys = {}
        existing_repos = releases_in_time_range[Release.repository_full_name.name].unique()
        physical_repos = coerce_logical_repos(existing_repos)

        # these matching rules must be applied to the past to stay consistent
        consistent_release_settings = ReleaseLoader.disambiguate_release_settings(
            release_settings, matched_bys)

        # seek releases backwards until each in_time_range one has a parent

        # preload releases not older than 5 weeks before `time_from`
        lookbehind_depth = time_from - timedelta(days=3 * 31)
        # if our initial guess failed, load releases until this exponential offset
        depth_step = timedelta(days=6 * 31)
        # we load all previous releases when we reach this depth
        # effectively, max 2 iterations
        lookbehind_depth_limit = time_from - timedelta(days=365)
        most_recent_time = time_from - timedelta(seconds=1)

        async def fetch_dags() -> Dict[str, DAG]:
            nonlocal pdags
            if pdags is None:
                pdags = await fetch_precomputed_commit_history_dags(
                    physical_repos, account, pdb, cache)
            return await fetch_repository_commits(
                pdags, releases_in_time_range, RELEASE_FETCH_COMMITS_COLUMNS, False,
                account, meta_ids, mdb, pdb, cache)

        if has_logical := contains_logical_repos(existing_repos):
            repo_name_origin_column = f"{Release.repository_full_name.name}_original"
            releases_in_time_range[repo_name_origin_column] = \
                releases_in_time_range[Release.repository_full_name.name]
            release_repos = releases_in_time_range[Release.repository_full_name.name].values
            for i, repo in enumerate(release_repos):
                release_repos[i] = drop_logical_repo(repo)

        async def dummy_load_releases_until_today() -> Tuple[pd.DataFrame, Any]:
            return dummy_releases_df(), None

        until_today_task = None
        if until_today:
            today = datetime.combine((datetime.now(timezone.utc) + timedelta(days=1)).date(),
                                     datetime.min.time(), tzinfo=timezone.utc)
            if today > time_to:
                until_today_task = cls.release_loader.load_releases(
                    # important: not existing_repos
                    repos, branches, default_branches, time_to, today,
                    consistent_release_settings, logical_settings, prefixer,
                    account, meta_ids, mdb, pdb, rdb, cache)
        if until_today_task is None:
            until_today_task = dummy_load_releases_until_today()

        (releases_today, _), (releases_previous, _), dags, repo_births = await gather(
            until_today_task,
            cls.release_loader.load_releases(
                existing_repos, branches, default_branches, lookbehind_depth, most_recent_time,
                consistent_release_settings, logical_settings,
                prefixer, account, meta_ids, mdb, pdb, rdb, cache),
            fetch_dags(),
            cls._fetch_first_release_dates(
                physical_repos, consistent_release_settings, default_branches,
                account, meta_ids, mdb, pdb, cache),
        )
        if extended_releases := [df for df in (releases_today, releases_previous) if not df.empty]:
            if len(extended_releases) == 2:
                extended_releases = pd.concat(extended_releases, ignore_index=True, copy=False)
            else:
                extended_releases = extended_releases[0]
            dags = await fetch_repository_commits(
                dags, extended_releases, RELEASE_FETCH_COMMITS_COLUMNS, False,
                account, meta_ids, mdb, pdb, cache)
        del extended_releases
        if has_logical:
            releases_in_time_range[Release.repository_full_name.name] = \
                releases_in_time_range[repo_name_origin_column]
            del releases_in_time_range[repo_name_origin_column]

        in_range_repos = releases_in_time_range[Release.repository_full_name.name].values
        in_range_shas = releases_in_time_range[Release.sha.name].values.astype("S40")
        in_range_dates = releases_in_time_range[Release.published_at.name].values

        not_enough_repos = [None]
        releases_previous_older = None
        while not_enough_repos:
            previous_shas = releases_previous[Release.sha.name].values.astype("S40")
            previous_dates = releases_previous[Release.published_at.name].values
            grouped_previous = dict(LogicalPRSettings.group_by_repo(
                releases_previous[Release.repository_full_name.name].values))
            not_enough_repos.clear()
            for repo, in_range_repo_indexes in LogicalPRSettings.group_by_repo(in_range_repos):
                physical_repo = drop_logical_repo(repo)
                if repo_births[physical_repo] >= lookbehind_depth:
                    continue
                previous_repo_indexes = grouped_previous.get(repo, [])
                if not len(previous_repo_indexes):
                    not_enough_repos.append(repo)
                    continue
                previous_repo_shas = previous_shas[previous_repo_indexes]
                previous_repo_dates = previous_dates[previous_repo_indexes]
                in_range_repo_shas = in_range_shas[in_range_repo_indexes]
                in_range_repo_dates = in_range_dates[in_range_repo_indexes]
                all_shas = np.concatenate([in_range_repo_shas, previous_repo_shas])
                all_timestamps = np.concatenate([in_range_repo_dates, previous_repo_dates])
                dag = dags[physical_repo]
                ownership = mark_dag_access(*dag, all_shas, True)
                parents = mark_dag_parents(*dag, all_shas, all_timestamps, ownership)
                if any((len(p) == 0) for p in parents[:len(in_range_repo_indexes)]):
                    not_enough_repos.append(repo)
            if not_enough_repos:
                most_recent_time = lookbehind_depth - timedelta(seconds=1)
                lookbehind_depth -= depth_step
                depth_step *= 2
                if lookbehind_depth < lookbehind_depth_limit:
                    lookbehind_depth = \
                        min(repo_births[drop_logical_repo(r)] for r in not_enough_repos)
                    if len(not_enough_repos) >= 3:
                        # load releases in `num_chunks` batches to save some resources
                        repo_birth_seq = sorted(
                            (repo_births[drop_logical_repo(r)], r) for r in not_enough_repos)
                        num_chunks = 3
                        step = len(not_enough_repos) // num_chunks + 1
                        chunks = await gather(*(cls.release_loader.load_releases(
                            [p[1] for p in repo_birth_seq[i * step:(i + 1) * step]],
                            branches, default_branches,
                            repo_birth_seq[min(i * step, len(repo_birth_seq) - 1)][0],
                            most_recent_time, consistent_release_settings, logical_settings,
                            prefixer, account, meta_ids, mdb, pdb, rdb, cache)
                            for i in range(num_chunks)))
                        releases_previous_older = pd.concat(
                            [c[0] for c in chunks], ignore_index=True, copy=False)
                if releases_previous_older is None:
                    releases_previous_older, _ = await cls.release_loader.load_releases(
                        not_enough_repos, branches, default_branches,
                        lookbehind_depth, most_recent_time,
                        consistent_release_settings, logical_settings,
                        prefixer, account, meta_ids, mdb, pdb, rdb, cache)
                dags = await fetch_repository_commits(
                    dags, releases_previous_older, RELEASE_FETCH_COMMITS_COLUMNS, False,
                    account, meta_ids, mdb, pdb, cache)
                releases_previous = pd.concat(
                    [releases_previous, releases_previous_older], ignore_index=True, copy=False)
                releases_previous_older = None

        releases = pd.concat([releases_today, releases_in_time_range, releases_previous],
                             ignore_index=True, copy=False)
        if not releases_today.empty:
            releases_in_time_range = pd.concat([releases_today, releases_in_time_range],
                                               ignore_index=True, copy=False)
        return matched_bys, releases, releases_in_time_range, consistent_release_settings, dags

    @classmethod
    @sentry_span
    async def _find_old_released_prs(cls,
                                     commits: np.ndarray,
                                     repos: np.ndarray,  # physical
                                     time_boundary: datetime,
                                     authors: Collection[str],
                                     mergers: Collection[str],
                                     jira: JIRAFilter,
                                     updated_min: Optional[datetime],
                                     updated_max: Optional[datetime],
                                     pr_blacklist: Optional[BinaryExpression],
                                     pr_whitelist: Optional[BinaryExpression],
                                     logical_settings: LogicalRepositorySettings,
                                     prefixer: Prefixer,
                                     meta_ids: Tuple[int, ...],
                                     mdb: Database,
                                     cache: Optional[aiomcache.Client],
                                     ) -> pd.DataFrame:
        assert len(commits) == len(repos)
        assert len(commits) > 0
        unique_physical_repos = {drop_logical_repo(r.decode()) for r in np.unique(repos)}
        if updated_min is not None:
            assert updated_max is not None
            # DEV-3730: load all the PRs and intersect merge commit shas with `commits`
            merge_commit_ids = None
        else:
            assert updated_max is None
            # performance: first find out the merge commit IDs, then use them to query PullRequest
            # instead of PullRequest.merge_commit_sha.in_(np.unique(commits).astype("U40")),
            # reliability: DEV-3333 requires us to skip non-existent repositories
            # according to how SQL works, `null` is ignored in `IN ('whatever', null, 'else')`
            repo_name_to_node = prefixer.repo_name_to_node.get
            filters = [
                NodeCommit.acc_id.in_(meta_ids),
                NodeCommit.repository_id.in_(
                    [repo_name_to_node(r) for r in unique_physical_repos]),
                NodeCommit.sha.in_(np.unique(commits).astype("U40")),
            ]
            commit_rows = await mdb.fetch_all(select([NodeCommit.node_id]).where(and_(*filters)))
            merge_commit_ids = [r[0] for r in commit_rows]
        filters = [
            PullRequest.merged_at < time_boundary,
            PullRequest.hidden.is_(False),
            PullRequest.acc_id.in_(meta_ids),
        ]
        if merge_commit_ids is not None:
            filters.append(PullRequest.merge_commit_id.in_(merge_commit_ids))
        else:
            filters.append(PullRequest.updated_at.between(updated_min, updated_max))
        if len(authors) and len(mergers):
            filters.append(or_(
                PullRequest.user_login.in_any_values(authors),
                PullRequest.merged_by_login.in_any_values(mergers),
            ))
        elif len(authors):
            filters.append(PullRequest.user_login.in_any_values(authors))
        elif len(mergers):
            filters.append(PullRequest.merged_by_login.in_any_values(mergers))
        if pr_blacklist is not None:
            filters.append(pr_blacklist)
        if pr_whitelist is not None:
            filters.append(pr_whitelist)
        if not jira:
            query = select([PullRequest]).where(and_(*filters))
        else:
            query = await generate_jira_prs_query(filters, jira, None, mdb, cache)
        query = query.order_by(PullRequest.merge_commit_sha.name)
        prs = await read_sql_query(query, mdb, PullRequest, index=PullRequest.node_id.name)
        if prs.empty:
            return prs
        if logical_settings.has_logical_prs():
            if logical_settings.has_prs_by_label():
                labels = await fetch_labels_to_filter(prs.index.values, meta_ids, mdb)
            else:
                labels = pd.DataFrame()
            prs = split_logical_prs(
                prs, labels,
                logical_settings.with_logical_prs(
                    prs[PullRequest.repository_full_name.name].unique()),
                logical_settings)
            prs.reset_index(PullRequest.repository_full_name.name, inplace=True)
        pr_commits = prs[PullRequest.merge_commit_sha.name].values.astype("S40")
        pr_repos = prs[PullRequest.repository_full_name.name].values.astype("S")
        mask = np.in1d(commits, pr_commits)
        commits = commits[mask]
        repos = repos[mask]
        pr_commit_repos = np.char.add(pr_commits, pr_repos)
        commit_repos = np.char.add(commits, repos)
        mask = np.in1d(pr_commit_repos, commit_repos, assume_unique=True)
        indexes = np.flatnonzero(mask)
        if len(indexes) < len(prs):
            prs = prs.take(indexes)
        return prs

    @classmethod
    @sentry_span
    @cached(
        exptime=middle_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda repos, release_settings, **_: (
            ",".join(sorted(repos)),
            release_settings,
        ),
        refresh_on_access=True,
    )
    async def _fetch_first_release_dates(cls,
                                         repos: Iterable[str],
                                         release_settings: ReleaseSettings,
                                         default_branches: Dict[str, str],
                                         account: int,
                                         meta_ids: Tuple[int, ...],
                                         mdb: Database,
                                         pdb: Database,
                                         cache: Optional[aiomcache.Client],
                                         ) -> Dict[str, datetime]:
        match_groups, _ = group_repos_by_release_match(repos, default_branches, release_settings)
        spans, earliest_releases, repo_biths = await gather(
            cls.release_loader.fetch_precomputed_release_match_spans(match_groups, account, pdb),
            cls._fetch_earliest_precomputed_releases(match_groups, account, pdb),
            cls._fetch_repository_first_commit_dates(repos, account, meta_ids, mdb, pdb, cache),
        )
        for key, val in repo_biths.items():
            try:
                span_from, span_to = spans[key][release_settings.native[key].match]
                earliest_release = earliest_releases[key] - timedelta(seconds=1)
            except KeyError:
                continue
            if val >= span_from and earliest_release < span_to:
                repo_biths[key] = earliest_release
        return repo_biths

    @classmethod
    @sentry_span
    async def _fetch_earliest_precomputed_releases(
            cls,
            match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
            account: int,
            pdb: Database) -> Dict[str, datetime]:
        prel = PrecomputedRelease
        or_items, _ = match_groups_to_sql(match_groups, prel)
        if not or_items:
            return {}
        query = union_all(*(
            select([prel.repository_full_name, func.min(prel.published_at)])
            .where(and_(item, prel.acc_id == account))
            .group_by(prel.repository_full_name)
            for item in or_items))
        result = dict(await pdb.fetch_all(query))
        if pdb.url.dialect == "sqlite":
            for key, val in result.items():
                result[key] = val.replace(tzinfo=timezone.utc)
        return result

    @classmethod
    @sentry_span
    @cached(
        exptime=max_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda meta_ids, repos, **_: (*meta_ids, ",".join(sorted(repos))),
        refresh_on_access=True,
    )
    async def _fetch_repository_first_commit_dates(cls,
                                                   repos: Iterable[str],
                                                   account: int,
                                                   meta_ids: Tuple[int, ...],
                                                   mdb: Database,
                                                   pdb: Database,
                                                   cache: Optional[aiomcache.Client],
                                                   ) -> Dict[str, datetime]:
        log = logging.getLogger(f"{metadata.__package__}._fetch_repository_first_commit_dates")
        rows = await pdb.fetch_all(
            select([GitHubRepository.repository_full_name,
                    GitHubRepository.first_commit.label("min")])
            .where(and_(GitHubRepository.repository_full_name.in_(repos),
                        GitHubRepository.acc_id == account)))
        add_pdb_hits(pdb, "_fetch_repository_first_commit_dates", len(rows))
        missing = set(repos) - {r[0] for r in rows}
        add_pdb_misses(pdb, "_fetch_repository_first_commit_dates", len(missing))
        if missing:
            computed = await mdb.fetch_all(
                select([func.min(NodeRepository.name_with_owner)
                        .label(PushCommit.repository_full_name.name),
                        func.min(NodeCommit.committed_date).label("min"),
                        NodeRepository.id])
                .select_from(join(NodeCommit, NodeRepository,
                                  and_(NodeCommit.repository_id == NodeRepository.id,
                                       NodeCommit.acc_id == NodeRepository.acc_id)))
                .where(and_(NodeRepository.name_with_owner.in_(missing),
                            NodeRepository.acc_id.in_(meta_ids)))
                .group_by(NodeRepository.id))
            values = []
            for r in computed:
                if (repo_name := r[PushCommit.repository_full_name.name]) in missing:
                    missing.remove(repo_name)
                    values.append(GitHubRepository(
                        acc_id=account,
                        repository_full_name=repo_name,
                        first_commit=r["min"],
                        node_id=r[NodeRepository.id.name],
                    ).create_defaults().explode(with_primary_keys=True))
                else:
                    dupes = [dr[NodeRepository.id.name] for dr in computed
                             if dr[PushCommit.repository_full_name.name] == repo_name]
                    log.error("duplicate repositories in %s@%s: %s -> %s",
                              NodeRepository.__tablename__, meta_ids, repo_name, dupes)
            if missing:
                log.warning("some repositories have 0 commits in %s: %s", meta_ids, missing)
                now = datetime.now(timezone.utc)
                for r in missing:
                    rows.append((r, now))
            if mdb.url.dialect == "sqlite":
                for v in values:
                    v[GitHubRepository.first_commit.name] = \
                        v[GitHubRepository.first_commit.name].replace(tzinfo=timezone.utc)
            await defer(insert_or_ignore(
                GitHubRepository, values, "_fetch_repository_first_commit_dates", pdb,
            ), "insert_repository_first_commit_dates")
            rows.extend((r[:2] for r in computed))
        result = dict(rows)
        if mdb.url.dialect == "sqlite" or pdb.url.dialect == "sqlite":
            for k, v in result.items():
                result[k] = v.replace(tzinfo=timezone.utc)
        return result

    @classmethod
    def _extract_released_commits(cls,
                                  releases: pd.DataFrame,
                                  dag: DAG,
                                  time_boundary: datetime,
                                  ) -> np.ndarray:
        time_mask = (releases[Release.published_at.name] >= time_boundary).values
        new_releases = releases.take(np.flatnonzero(time_mask))
        assert not new_releases.empty, "you must check this before calling me"
        hashes, vertexes, edges = dag
        visited_hashes, _, _ = extract_subdag(
            hashes, vertexes, edges, new_releases[Release.sha.name].values.astype("S40"))
        # we need to traverse the DAG from *all* the previous releases because of release branches
        if not time_mask.all():
            boundary_release_hashes = releases[Release.sha.name].values[~time_mask].astype("S40")
        else:
            boundary_release_hashes = []
        if len(boundary_release_hashes) == 0:
            return visited_hashes
        ignored_hashes, _, _ = extract_subdag(hashes, vertexes, edges, boundary_release_hashes)
        deleted_indexes = np.searchsorted(visited_hashes, ignored_hashes)
        # boundary_release_hash may touch some unique hashes not present in visited_hashes
        deleted_indexes = deleted_indexes[deleted_indexes < len(visited_hashes)]
        released_hashes = np.delete(visited_hashes, deleted_indexes)
        return released_hashes
