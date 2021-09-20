import asyncio
import bisect
from datetime import datetime, timedelta, timezone
import logging
import pickle
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple

import aiomcache
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, func, join, or_, select
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api import metadata
from athenian.api.async_utils import gather, postprocess_datetime, read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.branches import load_branch_commit_dates
from athenian.api.controllers.miners.github.commit import DAG, \
    fetch_precomputed_commit_history_dags, \
    fetch_repository_commits, RELEASE_FETCH_COMMITS_COLUMNS
from athenian.api.controllers.miners.github.dag_accelerated import extract_subdag, \
    mark_dag_access, searchsorted_inrange
from athenian.api.controllers.miners.github.precomputed_prs import \
    DonePRFactsLoader, MergedPRFactsLoader, update_unreleased_prs
from athenian.api.controllers.miners.github.release_load import dummy_releases_df, ReleaseLoader
from athenian.api.controllers.miners.github.released_pr import new_released_prs_df, release_columns
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.controllers.miners.types import nonemax, PullRequestFacts
from athenian.api.controllers.prefixer import PrefixerPromise
from athenian.api.controllers.settings import ReleaseMatch, ReleaseSettings
from athenian.api.db import add_pdb_hits, add_pdb_misses, insert_or_ignore, ParallelDatabase
from athenian.api.defer import defer
from athenian.api.models.metadata.github import NodeCommit, NodeRepository, PullRequest, \
    PullRequestLabel, PushCommit, Release
from athenian.api.models.precomputed.models import GitHubRepository
from athenian.api.tracing import sentry_span


async def load_commit_dags(releases: pd.DataFrame,
                           account: int,
                           meta_ids: Tuple[int, ...],
                           mdb: ParallelDatabase,
                           pdb: ParallelDatabase,
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
                                  prefixer: PrefixerPromise,
                                  account: int,
                                  meta_ids: Tuple[int, ...],
                                  mdb: ParallelDatabase,
                                  pdb: ParallelDatabase,
                                  cache: Optional[aiomcache.Client],
                                  ) -> Tuple[pd.DataFrame,
                                             Dict[str, Tuple[str, PullRequestFacts]],
                                             asyncio.Event]:
        """
        Match the merged pull requests to the nearest releases that include them.

        :return: 1. pd.DataFrame with the mapped PRs. \
                 2. Precomputed facts about unreleased merged PRs. \
                 3. Synchronization for updating the pdb table with merged unreleased PRs.
        """
        assert isinstance(time_to, datetime)
        assert isinstance(mdb, ParallelDatabase)
        assert isinstance(pdb, ParallelDatabase)
        pr_releases = new_released_prs_df()
        unreleased_prs_event = asyncio.Event()
        if prs.empty:
            unreleased_prs_event.set()
            return pr_releases, {}, unreleased_prs_event
        tasks = [
            load_branch_commit_dates(branches, meta_ids, mdb),
            MergedPRFactsLoader.load_merged_unreleased_pull_request_facts(
                prs, nonemax(releases[Release.published_at.name].nonemax(), time_to),
                LabelFilter.empty(), matched_bys, default_branches, release_settings,
                prefixer, account, pdb),
            DonePRFactsLoader.load_precomputed_pr_releases(
                prs.index, time_to, matched_bys, default_branches, release_settings,
                prefixer, account, pdb, cache),
        ]
        _, unreleased_prs, precomputed_pr_releases = await gather(*tasks)
        add_pdb_hits(pdb, "map_prs_to_releases/released", len(precomputed_pr_releases))
        add_pdb_hits(pdb, "map_prs_to_releases/unreleased", len(unreleased_prs))
        pr_releases = precomputed_pr_releases
        merged_prs = prs[~prs.index.isin(pr_releases.index.union(unreleased_prs))]
        if merged_prs.empty:
            unreleased_prs_event.set()
            return pr_releases, unreleased_prs, unreleased_prs_event
        tasks = [
            cls._fetch_labels(merged_prs.index, meta_ids, mdb),
            cls._map_prs_to_releases(merged_prs, dags, releases),
            cls._find_dead_merged_prs(merged_prs),
        ]
        labels, missed_released_prs, dead_prs = await gather(*tasks)
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
        releases = dict(list(releases.groupby(Release.repository_full_name.name, sort=False)))

        released_prs = []
        log = logging.getLogger("%s.map_prs_to_releases" % metadata.__package__)
        for repo, repo_prs in prs.groupby(PullRequest.repository_full_name.name, sort=False):
            try:
                repo_releases = releases[repo]
            except KeyError:
                # no releases exist for this repo
                continue
            repo_prs = repo_prs.take(
                np.where(~repo_prs[PullRequest.merge_commit_sha.name].isnull())[0])
            hashes, vertexes, edges = dags[repo]
            if len(hashes) == 0:
                log.error("very suspicious: empty DAG for %s\n%s", repo, repo_releases.to_csv())
            ownership = mark_dag_access(
                hashes, vertexes, edges, repo_releases[Release.sha.name].values.astype("S40"))
            unmatched = np.where(ownership == len(repo_releases))[0]
            if len(unmatched) > 0:
                hashes = np.delete(hashes, unmatched)
                ownership = np.delete(ownership, unmatched)
            if len(hashes) == 0:
                continue
            merge_hashes = repo_prs[PullRequest.merge_commit_sha.name].values.astype("S40")
            merges_found = searchsorted_inrange(hashes, merge_hashes)
            found_mask = hashes[merges_found] == merge_hashes
            found_releases = repo_releases[release_columns].take(
                ownership[merges_found[found_mask]])
            if not found_releases.empty:
                found_prs = repo_prs.index.take(np.nonzero(found_mask)[0])
                found_releases.set_index(found_prs, inplace=True)
                released_prs.append(found_releases)
            await asyncio.sleep(0)
        if released_prs:
            released_prs = pd.concat(released_prs, copy=False)
        else:
            released_prs = new_released_prs_df()
        released_prs[Release.published_at.name] = np.maximum(
            released_prs[Release.published_at.name],
            prs.loc[released_prs.index, PullRequest.merged_at.name])
        return postprocess_datetime(released_prs)

    @classmethod
    @sentry_span
    async def _find_dead_merged_prs(cls, prs: pd.DataFrame) -> pd.DataFrame:
        dead_indexes = np.flatnonzero(prs["dead"].values)
        dead_prs = [
            (pr_id, None, None, None, None, None, repo, ReleaseMatch.force_push_drop)
            for repo, pr_id in zip(
                prs[PullRequest.repository_full_name.name].take(dead_indexes).values,
                prs.index.take(dead_indexes).values)
        ]
        return new_released_prs_df(dead_prs)

    @classmethod
    @sentry_span
    async def _fetch_labels(cls,
                            node_ids: Iterable[str],
                            meta_ids: Tuple[int, ...],
                            mdb: ParallelDatabase,
                            ) -> Dict[str, List[str]]:
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
                                  updated_min: Optional[datetime],
                                  updated_max: Optional[datetime],
                                  pdags: Optional[Dict[str, DAG]],
                                  prefixer: PrefixerPromise,
                                  account: int,
                                  meta_ids: Tuple[int, ...],
                                  mdb: ParallelDatabase,
                                  pdb: ParallelDatabase,
                                  rdb: ParallelDatabase,
                                  cache: Optional[aiomcache.Client],
                                  pr_blacklist: Optional[BinaryExpression] = None,
                                  pr_whitelist: Optional[BinaryExpression] = None,
                                  truncate: bool = True,
                                  ) -> Tuple[pd.DataFrame,
                                             pd.DataFrame,
                                             Dict[str, ReleaseMatch],
                                             Dict[str, DAG]]:
        """Find pull requests which were released between `time_from` and `time_to` but merged before \
        `time_from`.

        :param authors: Required PR commit_authors.
        :param mergers: Required PR mergers.
        :param truncate: Do not load releases after `time_to`.
        :return: pd.DataFrame with found PRs that were created before `time_from` and released \
                 between `time_from` and `time_to` \
                 + \
                 pd.DataFrame with the discovered releases between \
                 `time_from` and `time_to` (today if not `truncate`) \
                 + \
                 `matched_bys` so that we don't have to compute that mapping again. \
                 + \
                 commit DAGs that contain the relevant releases.
        """
        assert isinstance(time_from, datetime)
        assert isinstance(time_to, datetime)
        assert isinstance(mdb, ParallelDatabase)
        assert isinstance(pdb, ParallelDatabase)
        assert isinstance(pr_blacklist, (BinaryExpression, type(None)))
        assert isinstance(pr_whitelist, (BinaryExpression, type(None)))
        assert (updated_min is None) == (updated_max is None)

        async def fetch_pdags():
            if pdags is None:
                return await fetch_precomputed_commit_history_dags(repos, account, pdb, cache)
            return pdags

        tasks = [
            cls._find_releases_for_matching_prs(
                repos, branches, default_branches, time_from, time_to, not truncate,
                release_settings, prefixer, account, meta_ids, mdb, pdb, rdb, cache),
            fetch_pdags(),
        ]
        (matched_bys, releases, releases_in_time_range, release_settings), pdags = await gather(
            *tasks)
        # ensure that our DAGs contain all the mentioned releases
        rpak = Release.published_at.name
        rrfnk = Release.repository_full_name.name
        dags = await fetch_repository_commits(
            pdags, releases, RELEASE_FETCH_COMMITS_COLUMNS, False, account, meta_ids,
            mdb, pdb, cache)
        all_observed_repos = []
        all_observed_commits = []
        # find the released commit hashes by two DAG traversals
        with sentry_sdk.start_span(op="_generate_released_prs_clause"):
            for repo, repo_releases in releases.groupby(rrfnk, sort=False):
                if (repo_releases[rpak] >= time_from).any():
                    observed_commits = cls._extract_released_commits(
                        repo_releases, dags[repo], time_from)
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
            prs = await cls._find_old_released_prs(
                all_observed_commits, all_observed_repos, time_from, authors, mergers, jira,
                updated_min, updated_max, pr_blacklist, pr_whitelist, meta_ids, mdb, cache)
        else:
            prs = pd.DataFrame(columns=[c.name for c in PullRequest.__table__.columns
                                        if c.name != PullRequest.node_id.name])
            prs.index = pd.Index([], name=PullRequest.node_id.name)
        prs["dead"] = False
        return prs, releases_in_time_range, matched_bys, dags

    @classmethod
    @sentry_span
    async def _find_releases_for_matching_prs(
            cls,
            repos: Iterable[str],
            branches: pd.DataFrame,
            default_branches: Dict[str, str],
            time_from: datetime,
            time_to: datetime,
            until_today: bool,
            release_settings: ReleaseSettings,
            prefixer: PrefixerPromise,
            account: int,
            meta_ids: Tuple[int, ...],
            mdb: ParallelDatabase,
            pdb: ParallelDatabase,
            rdb: ParallelDatabase,
            cache: Optional[aiomcache.Client],
            releases_in_time_range: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict[str, ReleaseMatch],
               pd.DataFrame,
               pd.DataFrame,
               ReleaseSettings]:
        """
        Load releases with sufficient history depth.

        1. Load releases between `time_from` and `time_to`, record the effective release matches.
        2. Use those matches to load enough releases before `time_from` to ensure we don't get \
           "release leakages" in the commit DAG. Ideally, we should use the DAGs, but we take \
           risks and just set a long enough lookbehind time interval.
        3. Optionally, use those matches to load all the releases after `time_to`.
        """
        if releases_in_time_range is None:
            # we have to load releases in two separate batches: before and after time_from
            # that's because the release strategy can change depending on the time range
            # see ENG-710 and ENG-725
            releases_in_time_range, matched_bys = await cls.release_loader.load_releases(
                repos, branches, default_branches, time_from, time_to,
                release_settings, prefixer, account, meta_ids, mdb, pdb, rdb, cache)
        else:
            matched_bys = {}
        # these matching rules must be applied in the past to stay consistent
        consistent_release_settings = ReleaseLoader.disambiguate_release_settings(
            release_settings, matched_bys)
        repos_matched_by_tag = []
        repos_matched_by_branch = []
        for repo in repos:
            match = consistent_release_settings.native[repo].match
            if match in (ReleaseMatch.tag, ReleaseMatch.event):
                repos_matched_by_tag.append(repo)
            elif match == ReleaseMatch.branch:
                repos_matched_by_branch.append(repo)

        async def dummy_load_releases_until_today() -> Tuple[pd.DataFrame, Any]:
            return dummy_releases_df(), None

        until_today_task = None
        if until_today:
            today = datetime.combine((datetime.now(timezone.utc) + timedelta(days=1)).date(),
                                     datetime.min.time(), tzinfo=timezone.utc)
            if today > time_to:
                until_today_task = cls.release_loader.load_releases(
                    repos, branches, default_branches, time_to, today,
                    consistent_release_settings, prefixer,
                    account, meta_ids, mdb, pdb, rdb, cache)
        if until_today_task is None:
            until_today_task = dummy_load_releases_until_today()

        # there are two groups of repos now: matched by tag and by branch
        # we have to fetch *all* the tags from the past because:
        # some repos fork a new branch for each release and make a unique release commit
        # some repos maintain several major versions in parallel
        # so when somebody releases 1.1.0 in August 2020 alongside with 2.0.0 released in June 2020
        # and 1.0.0 in September 2018, we must load 1.0.0, otherwise the PR for 1.0.0 release
        # will be matched to 1.1.0 in August 2020 and will have a HUGE release time

        # we are golden if we match by branch, one older merge preceding `time_from` should be fine
        # unless there are several release branches; we hope for the best then
        # so we split repos and take two different logic paths

        # find branch releases not older than 5 weeks before `time_from`
        branch_lookbehind_time_from = time_from - timedelta(days=5 * 7)
        # find tag releases not older than 2 years before `time_from`
        tag_lookbehind_time_from = time_from - timedelta(days=2 * 365)
        # look for releases up till this date
        most_recent_time = time_from - timedelta(seconds=1)
        tasks = [
            until_today_task,
            cls.release_loader.load_releases(repos_matched_by_branch, branches, default_branches,
                                             branch_lookbehind_time_from, most_recent_time,
                                             consistent_release_settings, prefixer, account,
                                             meta_ids, mdb, pdb, rdb, cache),
            cls.release_loader.load_releases(repos_matched_by_tag, branches, default_branches,
                                             tag_lookbehind_time_from, most_recent_time,
                                             consistent_release_settings, prefixer, account,
                                             meta_ids, mdb, pdb, rdb, cache),
            cls._fetch_repository_first_commit_dates(repos_matched_by_branch, account, meta_ids,
                                                     mdb, pdb, cache),
        ]
        releases_today, releases_old_branches, releases_old_tags, repo_births = await gather(
            *tasks)
        releases_today = releases_today[0]
        releases_old_branches = releases_old_branches[0]
        releases_old_tags = releases_old_tags[0]
        hard_repos = set(repos_matched_by_branch) - \
            set(releases_old_branches[Release.repository_full_name.name].unique())
        if hard_repos:
            with sentry_sdk.start_span(op="_find_releases_for_matching_prs/hard_repos"):
                repo_births = sorted((v, k) for k, v in repo_births.items() if k in hard_repos)
                repo_births_dates = [rb[0].replace(tzinfo=timezone.utc) for rb in repo_births]
                repo_births_names = [rb[1] for rb in repo_births]
                del repo_births
                deeper_step = timedelta(days=6 * 31)
                while hard_repos:
                    # no previous releases were discovered for `hard_repos`, go deeper in history
                    hard_repos = hard_repos.intersection(repo_births_names[:bisect.bisect_right(
                        repo_births_dates, branch_lookbehind_time_from)])
                    if not hard_repos:
                        break
                    extra_releases, _ = await cls.release_loader.load_releases(
                        hard_repos, branches, default_branches,
                        branch_lookbehind_time_from - deeper_step, branch_lookbehind_time_from,
                        consistent_release_settings, prefixer, account, meta_ids,
                        mdb, pdb, rdb, cache)
                    releases_old_branches = releases_old_branches.append(extra_releases)
                    hard_repos -= set(extra_releases[Release.repository_full_name.name].unique())
                    del extra_releases
                    branch_lookbehind_time_from -= deeper_step
                    deeper_step *= 2
        releases = pd.concat([releases_today, releases_in_time_range,
                              releases_old_branches, releases_old_tags],
                             ignore_index=True, copy=False)
        releases.sort_values(Release.published_at.name,
                             inplace=True, ascending=False, ignore_index=True)
        if not releases_today.empty:
            releases_in_time_range = pd.concat([releases_today, releases_in_time_range],
                                               ignore_index=True, copy=False)
        return matched_bys, releases, releases_in_time_range, consistent_release_settings

    @classmethod
    @sentry_span
    async def _find_old_released_prs(cls,
                                     commits: np.ndarray,
                                     repos: np.ndarray,
                                     time_boundary: datetime,
                                     authors: Collection[str],
                                     mergers: Collection[str],
                                     jira: JIRAFilter,
                                     updated_min: Optional[datetime],
                                     updated_max: Optional[datetime],
                                     pr_blacklist: Optional[BinaryExpression],
                                     pr_whitelist: Optional[BinaryExpression],
                                     meta_ids: Tuple[int, ...],
                                     mdb: ParallelDatabase,
                                     cache: Optional[aiomcache.Client],
                                     ) -> pd.DataFrame:
        assert len(commits) == len(repos)
        assert len(commits) > 0
        filters = [
            PullRequest.merged_at < time_boundary,
            PullRequest.hidden.is_(False),
            PullRequest.acc_id.in_(meta_ids),
            PullRequest.merge_commit_sha.in_(commits.astype("U40")),
        ]
        if updated_min is not None:
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
            query = await generate_jira_prs_query(filters, jira, mdb, cache)
        query = query.order_by(PullRequest.merge_commit_sha.name)
        prs = await read_sql_query(query, mdb, PullRequest, index=PullRequest.node_id.name)
        if prs.empty:
            return prs
        pr_commits = prs[PullRequest.merge_commit_sha.name].values.astype("S40")
        pr_repos = prs[PullRequest.repository_full_name.name].values.astype("S")
        indexes = np.searchsorted(commits, pr_commits)
        checked = np.nonzero(pr_repos == repos[indexes])[0]
        if len(checked) < len(prs):
            prs = prs.take(checked)
        return prs

    @classmethod
    @sentry_span
    @cached(
        exptime=24 * 60 * 60,  # 1 day
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda repos, **_: (",".join(sorted(repos)),),
        refresh_on_access=True,
    )
    async def _fetch_repository_first_commit_dates(cls,
                                                   repos: Iterable[str],
                                                   account: int,
                                                   meta_ids: Tuple[int, ...],
                                                   mdb: ParallelDatabase,
                                                   pdb: ParallelDatabase,
                                                   cache: Optional[aiomcache.Client],
                                                   ) -> Dict[str, datetime]:
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
            if computed:
                values = [
                    GitHubRepository(
                        acc_id=account,
                        repository_full_name=r[0],
                        first_commit=r[1],
                        node_id=r[2],
                    ).create_defaults().explode(with_primary_keys=True)
                    for r in computed
                ]
                if mdb.url.dialect == "sqlite":
                    for v in values:
                        v[GitHubRepository.first_commit.name] = \
                            v[GitHubRepository.first_commit.name].replace(tzinfo=timezone.utc)
                await defer(insert_or_ignore(
                    GitHubRepository, values, "_fetch_repository_first_commit_dates", pdb,
                ), "insert_repository_first_commit_dates")
                rows.extend(computed)
        result = {r[0]: r[1] for r in rows}
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
        time_mask = releases[Release.published_at.name] >= time_boundary
        new_releases = releases.take(np.where(time_mask)[0])
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
