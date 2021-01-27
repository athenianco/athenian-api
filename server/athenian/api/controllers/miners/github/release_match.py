import asyncio
import bisect
from datetime import datetime, timedelta, timezone
import logging
import pickle
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple

import aiomcache
import databases
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, func, insert, join, or_, select
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api import metadata
from athenian.api.async_utils import gather, postprocess_datetime, read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.commit import BRANCH_FETCH_COMMITS_COLUMNS, \
    fetch_precomputed_commit_history_dags, \
    fetch_repository_commits, RELEASE_FETCH_COMMITS_COLUMNS
from athenian.api.controllers.miners.github.dag_accelerated import extract_subdag, \
    mark_dag_access, searchsorted_inrange
from athenian.api.controllers.miners.github.precomputed_prs import \
    load_merged_unreleased_pull_request_facts, load_precomputed_pr_releases, \
    update_unreleased_prs
from athenian.api.controllers.miners.github.release_load import dummy_releases_df, load_releases
from athenian.api.controllers.miners.github.released_pr import matched_by_column, \
    new_released_prs_df
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.controllers.miners.types import nonemax, PullRequestFacts
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.db import add_pdb_hits, add_pdb_misses
from athenian.api.defer import defer
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import Branch, NodeCommit, NodeRepository, PullRequest, \
    PullRequestLabel, PushCommit, Release
from athenian.api.models.precomputed.models import GitHubRepository
from athenian.api.tracing import sentry_span


@sentry_span
async def map_prs_to_releases(prs: pd.DataFrame,
                              releases: pd.DataFrame,
                              matched_bys: Dict[str, ReleaseMatch],
                              branches: pd.DataFrame,
                              default_branches: Dict[str, str],
                              time_to: datetime,
                              dags: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                              release_settings: Dict[str, ReleaseMatchSetting],
                              meta_ids: Tuple[int, ...],
                              mdb: databases.Database,
                              pdb: databases.Database,
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
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    pr_releases = new_released_prs_df()
    unreleased_prs_event = asyncio.Event()
    if prs.empty:
        unreleased_prs_event.set()
        return pr_releases, {}, unreleased_prs_event
    branch_commit_ids = branches[Branch.commit_id.key].values
    tasks = [
        mdb.fetch_all(select([NodeCommit.id, NodeCommit.committed_date])
                      .where(and_(NodeCommit.id.in_(branch_commit_ids),
                                  NodeCommit.acc_id.in_(meta_ids)))),
        load_merged_unreleased_pull_request_facts(
            prs, nonemax(releases[Release.published_at.key].nonemax(), time_to),
            LabelFilter.empty(), matched_bys, default_branches, release_settings, pdb),
        load_precomputed_pr_releases(
            prs.index, time_to, matched_bys, default_branches, release_settings, pdb, cache),
    ]
    branch_commit_dates, unreleased_prs, precomputed_pr_releases = await gather(*tasks)
    add_pdb_hits(pdb, "map_prs_to_releases/released", len(precomputed_pr_releases))
    add_pdb_hits(pdb, "map_prs_to_releases/unreleased", len(unreleased_prs))
    pr_releases = precomputed_pr_releases
    merged_prs = prs[~prs.index.isin(pr_releases.index.union(unreleased_prs))]
    if merged_prs.empty:
        unreleased_prs_event.set()
        return pr_releases, unreleased_prs, unreleased_prs_event
    branch_commit_dates = {r[0]: r[1] for r in branch_commit_dates}
    if mdb.url.dialect == "sqlite":
        branch_commit_dates = {k: v.replace(tzinfo=timezone.utc)
                               for k, v in branch_commit_dates.items()}
    now = datetime.now(timezone.utc)
    branches[Branch.commit_date] = [branch_commit_dates.get(commit_id, now)
                                    for commit_id in branch_commit_ids]
    tasks = [
        _fetch_labels(merged_prs.index, meta_ids, mdb),
        _find_dead_merged_prs(merged_prs, dags, branches, meta_ids, mdb, pdb, cache),
        _map_prs_to_releases(merged_prs, dags, releases),
    ]
    labels, dead_prs, missed_released_prs = await gather(*tasks)
    # PRs may wrongly classified as dead although they are really released; remove the conflicts
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
        release_settings, pdb, unreleased_prs_event),
        "update_unreleased_prs(%d, %d)" % (len(merged_prs), len(missed_released_prs)))
    return pr_releases.append(missed_released_prs), unreleased_prs, unreleased_prs_event


async def _map_prs_to_releases(prs: pd.DataFrame,
                               dags: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                               releases: pd.DataFrame,
                               ) -> pd.DataFrame:
    if prs.empty:
        return new_released_prs_df()
    releases = dict(list(releases.groupby(Release.repository_full_name.key, sort=False)))

    released_prs = []
    release_columns = [
        c.key for c in (Release.published_at, Release.author, Release.url,
                        Release.id, Release.repository_full_name)
    ] + [matched_by_column]
    log = logging.getLogger("%s.map_prs_to_releases" % metadata.__package__)
    for repo, repo_prs in prs.groupby(PullRequest.repository_full_name.key, sort=False):
        try:
            repo_releases = releases[repo]
        except KeyError:
            # no releases exist for this repo
            continue
        repo_prs = repo_prs.take(np.where(~repo_prs[PullRequest.merge_commit_sha.key].isnull())[0])
        hashes, vertexes, edges = dags[repo]
        if len(hashes) == 0:
            log.error("Very suspicious: empty DAG for %s\n%s",
                      repo, repo_releases.to_csv())
        ownership = mark_dag_access(hashes, vertexes, edges, repo_releases[Release.sha.key].values)
        unmatched = np.where(ownership == len(repo_releases))[0]
        if len(unmatched) > 0:
            hashes = np.delete(hashes, unmatched)
            ownership = np.delete(ownership, unmatched)
        if len(hashes) == 0:
            continue
        merge_hashes = repo_prs[PullRequest.merge_commit_sha.key].values.astype("U40")
        merges_found = searchsorted_inrange(hashes, merge_hashes)
        found_mask = hashes[merges_found] == merge_hashes
        found_releases = repo_releases[release_columns].take(ownership[merges_found[found_mask]])
        if not found_releases.empty:
            found_prs = repo_prs.index.take(np.where(found_mask)[0])
            found_releases.set_index(found_prs, inplace=True)
            released_prs.append(found_releases)
        await asyncio.sleep(0)
    if released_prs:
        released_prs = pd.concat(released_prs, copy=False)
    else:
        released_prs = new_released_prs_df()
    released_prs[Release.published_at.key] = np.maximum(
        released_prs[Release.published_at.key],
        prs.loc[released_prs.index, PullRequest.merged_at.key])
    return postprocess_datetime(released_prs)


@sentry_span
async def _find_dead_merged_prs(prs: pd.DataFrame,
                                dags: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                branches: pd.DataFrame,
                                meta_ids: Tuple[int, ...],
                                mdb: databases.Database,
                                pdb: databases.Database,
                                cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    if branches.empty:
        return new_released_prs_df()
    prs = prs.take(np.where(
        prs[PullRequest.merged_at.key] <= datetime.now(timezone.utc) - timedelta(hours=1))[0])
    # timedelta(hours=1) must match the `exptime` of `fetch_repository_commits()`
    # commits DAGs are cached and may be not fully up to date, so otherwise some PRs may appear in
    # dead_prs and missed_released_prs at the same time
    # see also: DEV-554
    if prs.empty:
        return new_released_prs_df()
    rfnkey = PullRequest.repository_full_name.key
    mchkey = PullRequest.merge_commit_sha.key
    dead_prs = []
    dags = await fetch_repository_commits(
        dags, branches, BRANCH_FETCH_COMMITS_COLUMNS, True, meta_ids, mdb, pdb, cache)
    with sentry_sdk.start_span(op="_find_dead_merged_prs/search"):
        for repo, repo_prs in prs[[mchkey, rfnkey]].groupby(rfnkey, sort=False):
            hashes, _, _ = dags[repo]
            if len(hashes) == 0:
                # no branches found in `fetch_repository_commits()`
                continue
            pr_merge_hashes = repo_prs[mchkey].values.astype("U40")
            indexes = searchsorted_inrange(hashes, pr_merge_hashes)
            dead_indexes = np.where(pr_merge_hashes != hashes[indexes])[0]
            dead_prs.extend((pr_id, None, None, None, None, repo, ReleaseMatch.force_push_drop)
                            for pr_id in repo_prs.index.values[dead_indexes])
    return new_released_prs_df(dead_prs)


@sentry_span
async def _fetch_labels(node_ids: Iterable[str],
                        meta_ids: Tuple[int, ...],
                        mdb: databases.Database,
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


async def load_commit_dags(releases: pd.DataFrame,
                           meta_ids: Tuple[int, ...],
                           mdb: databases.Database,
                           pdb: databases.Database,
                           cache: Optional[aiomcache.Client],
                           ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Produce the commit history DAGs which should contain the specified releases."""
    pdags = await fetch_precomputed_commit_history_dags(
        releases[Release.repository_full_name.key].unique(), pdb, cache)
    return await fetch_repository_commits(
        pdags, releases, RELEASE_FETCH_COMMITS_COLUMNS, False, meta_ids, mdb, pdb, cache)


@sentry_span
async def _find_old_released_prs(commits: np.ndarray,
                                 repos: np.ndarray,
                                 time_boundary: datetime,
                                 authors: Collection[str],
                                 mergers: Collection[str],
                                 jira: JIRAFilter,
                                 updated_min: Optional[datetime],
                                 updated_max: Optional[datetime],
                                 pr_blacklist: Optional[BinaryExpression],
                                 meta_ids: Tuple[int, ...],
                                 mdb: databases.Database,
                                 cache: Optional[aiomcache.Client],
                                 ) -> pd.DataFrame:
    assert len(commits) == len(repos)
    assert len(commits) > 0
    filters = [
        PullRequest.merged_at < time_boundary,
        PullRequest.hidden.is_(False),
        PullRequest.acc_id.in_(meta_ids),
        PullRequest.merge_commit_sha.in_(commits),
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
    if not jira:
        query = select([PullRequest]).where(and_(*filters))
    else:
        query = await generate_jira_prs_query(filters, jira, mdb, cache)
    query = query.order_by(PullRequest.merge_commit_sha.key)
    prs = await read_sql_query(query, mdb, PullRequest, index=PullRequest.node_id.key)
    if prs.empty:
        return prs
    pr_commits = prs[PullRequest.merge_commit_sha.key].values
    pr_repos = prs[PullRequest.repository_full_name.key].values
    indexes = np.searchsorted(commits, pr_commits)
    checked = np.nonzero(pr_repos == repos[indexes])[0]
    if len(checked) < len(prs):
        prs = prs.take(checked)
    return prs


def _extract_released_commits(releases: pd.DataFrame,
                              dag: Tuple[np.ndarray, np.ndarray, np.ndarray],
                              time_boundary: datetime,
                              ) -> np.ndarray:
    time_mask = releases[Release.published_at.key] >= time_boundary
    new_releases = releases.take(np.where(time_mask)[0])
    assert not new_releases.empty, "you must check this before calling me"
    hashes, vertexes, edges = dag
    visited_hashes, _, _ = extract_subdag(
        hashes, vertexes, edges, new_releases[Release.sha.key].values.astype("U40"))
    # we need to traverse the DAG from *all* the previous releases because of release branches
    if not time_mask.all():
        boundary_release_hashes = releases[Release.sha.key].values[~time_mask].astype("U40")
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


@sentry_span
async def map_releases_to_prs(repos: Collection[str],
                              branches: pd.DataFrame,
                              default_branches: Dict[str, str],
                              time_from: datetime,
                              time_to: datetime,
                              authors: Collection[str],
                              mergers: Collection[str],
                              jira: JIRAFilter,
                              release_settings: Dict[str, ReleaseMatchSetting],
                              updated_min: Optional[datetime],
                              updated_max: Optional[datetime],
                              meta_ids: Tuple[int, ...],
                              mdb: databases.Database,
                              pdb: databases.Database,
                              cache: Optional[aiomcache.Client],
                              pr_blacklist: Optional[BinaryExpression] = None,
                              truncate: bool = True,
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, ReleaseMatch],
                                         Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
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
             `matched_bys` so that we don't have to compute that mapping again.
    """
    assert isinstance(time_from, datetime)
    assert isinstance(time_to, datetime)
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    assert isinstance(pr_blacklist, (BinaryExpression, type(None)))
    assert (updated_min is None) == (updated_max is None)

    tasks = [
        _find_releases_for_matching_prs(
            repos, branches, default_branches, time_from, time_to,
            not truncate, release_settings, meta_ids, mdb, pdb, cache),
        fetch_precomputed_commit_history_dags(repos, pdb, cache),
    ]
    (matched_bys, releases, releases_in_time_range, release_settings), pdags = await gather(*tasks)

    # ensure that our DAGs contain all the mentioned releases
    rpak = Release.published_at.key
    rrfnk = Release.repository_full_name.key
    dags = await fetch_repository_commits(
        pdags, releases, RELEASE_FETCH_COMMITS_COLUMNS, False, meta_ids, mdb, pdb, cache)
    all_observed_repos = []
    all_observed_commits = []
    # find the released commit hashes by two DAG traversals
    with sentry_sdk.start_span(op="_generate_released_prs_clause"):
        for repo, repo_releases in releases.groupby(rrfnk, sort=False):
            if (repo_releases[rpak] >= time_from).any():
                observed_commits = _extract_released_commits(repo_releases, dags[repo], time_from)
                if len(observed_commits):
                    all_observed_commits.append(observed_commits)
                    all_observed_repos.append(np.full_like(observed_commits, repo))
    if all_observed_commits:
        all_observed_repos = np.concatenate(all_observed_repos)
        all_observed_commits = np.concatenate(all_observed_commits)
        order = np.argsort(all_observed_commits)
        all_observed_commits = all_observed_commits[order]
        all_observed_repos = all_observed_repos[order]
        prs = await _find_old_released_prs(
            all_observed_commits, all_observed_repos, time_from, authors, mergers, jira,
            updated_min, updated_max, pr_blacklist, meta_ids, mdb, cache)
    else:
        prs = pd.DataFrame(columns=[c.name for c in PullRequest.__table__.columns
                                    if c.name != PullRequest.node_id.key])
        prs.index = pd.Index([], name=PullRequest.node_id.key)
    return prs, releases_in_time_range, matched_bys, dags


@sentry_span
async def _find_releases_for_matching_prs(repos: Iterable[str],
                                          branches: pd.DataFrame,
                                          default_branches: Dict[str, str],
                                          time_from: datetime,
                                          time_to: datetime,
                                          until_today: bool,
                                          release_settings: Dict[str, ReleaseMatchSetting],
                                          meta_ids: Tuple[int, ...],
                                          mdb: databases.Database,
                                          pdb: databases.Database,
                                          cache: Optional[aiomcache.Client],
                                          releases_in_time_range: Optional[pd.DataFrame] = None,
                                          ) -> Tuple[Dict[str, ReleaseMatch],
                                                     pd.DataFrame,
                                                     pd.DataFrame,
                                                     Dict[str, ReleaseMatchSetting]]:
    """
    Load releases with sufficient history depth.

    1. Load releases between `time_from` and `time_to`, record the effective release matches.
    2. Use those matches to load enough releases before `time_from` to ensure we don't get \
       "release leakages" in the commit DAG. Ideally, we should use the DAGs, but we take risks \
       and just set a long enough lookbehind time interval.
    3. Optionally, use those matches to load all the releases after `time_to`.
    """
    if releases_in_time_range is None:
        # we have to load releases in two separate batches: before and after time_from
        # that's because the release strategy can change depending on the time range
        # see ENG-710 and ENG-725
        releases_in_time_range, matched_bys = await load_releases(
            repos, branches, default_branches, time_from, time_to,
            release_settings, meta_ids, mdb, pdb, cache)
    else:
        matched_bys = {}
    # these matching rules must be applied in the past to stay consistent
    prefix = PREFIXES["github"]
    consistent_release_settings = {}
    repos_matched_by_tag = []
    repos_matched_by_branch = []
    for repo in repos:
        setting = release_settings[prefix + repo]
        match = ReleaseMatch(matched_bys.setdefault(repo, setting.match))
        consistent_release_settings[prefix + repo] = ReleaseMatchSetting(
            tags=setting.tags,
            branches=setting.branches,
            match=match,
        )
        if match == ReleaseMatch.tag:
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
            until_today_task = load_releases(
                repos, branches, default_branches, time_to, today,
                consistent_release_settings, meta_ids, mdb, pdb, cache)
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
    tasks = [
        until_today_task,
        load_releases(repos_matched_by_branch, branches, default_branches,
                      branch_lookbehind_time_from, time_from, consistent_release_settings,
                      meta_ids, mdb, pdb, cache),
        load_releases(repos_matched_by_tag, branches, default_branches,
                      tag_lookbehind_time_from, time_from, consistent_release_settings,
                      meta_ids, mdb, pdb, cache),
        _fetch_repository_first_commit_dates(repos_matched_by_branch, meta_ids, mdb, pdb, cache),
    ]
    releases_today, releases_old_branches, releases_old_tags, repo_births = await gather(*tasks)
    releases_today = releases_today[0]
    releases_old_branches = releases_old_branches[0]
    releases_old_tags = releases_old_tags[0]
    hard_repos = set(repos_matched_by_branch) - \
        set(releases_old_branches[Release.repository_full_name.key].unique())
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
                extra_releases, _ = await load_releases(
                    hard_repos, branches, default_branches,
                    branch_lookbehind_time_from - deeper_step, branch_lookbehind_time_from,
                    consistent_release_settings, meta_ids, mdb, pdb, cache)
                releases_old_branches = releases_old_branches.append(extra_releases)
                hard_repos -= set(extra_releases[Release.repository_full_name.key].unique())
                del extra_releases
                branch_lookbehind_time_from -= deeper_step
                deeper_step *= 2
    releases = pd.concat([releases_today, releases_in_time_range,
                          releases_old_branches, releases_old_tags],
                         ignore_index=True, copy=False)
    releases.sort_values(Release.published_at.key,
                         inplace=True, ascending=False, ignore_index=True)
    if not releases_today.empty:
        releases_in_time_range = pd.concat([releases_today, releases_in_time_range],
                                           ignore_index=True, copy=False)
    return matched_bys, releases, releases_in_time_range, consistent_release_settings


@sentry_span
@cached(
    exptime=24 * 60 * 60,  # 1 day
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, **_: (",".join(sorted(repos)),),
    refresh_on_access=True,
)
async def _fetch_repository_first_commit_dates(repos: Iterable[str],
                                               meta_ids: Tuple[int, ...],
                                               mdb: databases.Database,
                                               pdb: databases.Database,
                                               cache: Optional[aiomcache.Client],
                                               ) -> Dict[str, datetime]:
    rows = await pdb.fetch_all(
        select([GitHubRepository.repository_full_name,
                GitHubRepository.first_commit.label("min")])
        .where(GitHubRepository.repository_full_name.in_(repos)))
    add_pdb_hits(pdb, "_fetch_repository_first_commit_dates", len(rows))
    missing = set(repos) - {r[0] for r in rows}
    add_pdb_misses(pdb, "_fetch_repository_first_commit_dates", len(missing))
    if missing:
        computed = await mdb.fetch_all(
            select([func.min(NodeRepository.name_with_owner)
                    .label(PushCommit.repository_full_name.key),
                    func.min(NodeCommit.committed_date).label("min"),
                    NodeRepository.id])
            .select_from(join(NodeCommit, NodeRepository,
                              and_(NodeCommit.repository == NodeRepository.id,
                                   NodeCommit.acc_id == NodeRepository.acc_id)))
            .where(and_(NodeRepository.name_with_owner.in_(missing),
                        NodeRepository.acc_id.in_(meta_ids)))
            .group_by(NodeRepository.id))
        if computed:
            values = [GitHubRepository(repository_full_name=r[0], first_commit=r[1], node_id=r[2])
                      .create_defaults().explode(with_primary_keys=True)
                      for r in computed]
            if mdb.url.dialect == "sqlite":
                for v in values:
                    v[GitHubRepository.first_commit.key] = \
                        v[GitHubRepository.first_commit.key].replace(tzinfo=timezone.utc)

            async def insert_repository():
                async with pdb.connection() as pdb_conn:
                    async with pdb_conn.transaction():
                        try:
                            await pdb_conn.execute_many(insert(GitHubRepository), values)
                        except Exception as e:
                            log = logging.getLogger(
                                "%s._fetch_repository_first_commit_dates" % metadata.__package__)
                            log.warning("Failed to store %d rows: %s: %s",
                                        len(values), type(e).__name__, e)
            await defer(insert_repository(), "insert_repository")
            rows.extend(computed)
    result = {r[0]: r[1] for r in rows}
    if mdb.url.dialect == "sqlite" or pdb.url.dialect == "sqlite":
        for k, v in result.items():
            result[k] = v.replace(tzinfo=timezone.utc)
    return result
