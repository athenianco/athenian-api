import asyncio
import bisect
from datetime import datetime, timedelta, timezone
from io import BytesIO
from itertools import chain
import logging
import marshal
import pickle
import re
from typing import Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import aiomcache
import asyncpg
import databases
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, desc, distinct, func, insert, join, or_, select
from sqlalchemy.cprocessors import str_to_datetime
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api import metadata
from athenian.api.async_read_sql_query import postprocess_datetime, read_sql_query, wrap_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.github.precomputed_prs import discover_unreleased_prs, \
    load_precomputed_pr_releases, update_unreleased_prs
from athenian.api.controllers.miners.github.release_accelerated import update_history
from athenian.api.controllers.miners.github.released_pr import matched_by_column, \
    new_released_prs_df
from athenian.api.controllers.settings import default_branch_alias, ReleaseMatch, \
    ReleaseMatchSetting
from athenian.api.db import add_pdb_hits, add_pdb_misses
from athenian.api.defer import defer
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import Branch, NodeCommit, NodeRepository, PullRequest, \
    PullRequestLabel, PushCommit, Release, User
from athenian.api.models.precomputed.models import GitHubCommitFirstParents, GitHubCommitHistory, \
    GitHubRepository, GitHubRepositoryCommits
from athenian.api.tracing import sentry_span


@sentry_span
async def load_releases(repos: Iterable[str],
                        branches: pd.DataFrame,
                        default_branches: Dict[str, str],
                        time_from: datetime,
                        time_to: datetime,
                        settings: Dict[str, ReleaseMatchSetting],
                        mdb: databases.Database,
                        pdb: databases.Database,
                        cache: Optional[aiomcache.Client],
                        index: Optional[Union[str, Sequence[str]]] = None,
                        ) -> Tuple[pd.DataFrame, Dict[str, ReleaseMatch]]:
    """
    Fetch releases from the metadata DB according to the match settings.

    :param repos: Repositories in which to search for releases *without the service prefix*.
    :param branches: DataFrame with all the branches in `repos`.
    :param default_branches: Mapping from repository name to default branch name.
    """
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    assert time_from <= time_to
    repos_by_tag = []
    repos_by_tag_or_branch = []
    repos_by_branch = []
    prefix = PREFIXES["github"]
    for repo in repos:
        v = settings[prefix + repo]
        if v.match == ReleaseMatch.tag:
            repos_by_tag.append(repo)
        elif v.match == ReleaseMatch.tag_or_branch:
            repos_by_tag_or_branch.append(repo)
        elif v.match == ReleaseMatch.branch:
            repos_by_branch.append(repo)
    result = []
    applied_matches = {}
    if repos_by_tag:
        result.append(_match_releases_by_tag(
            repos_by_tag, time_from, time_to, settings, mdb))
        for k in repos_by_tag:
            applied_matches[k] = ReleaseMatch.tag
    if repos_by_tag_or_branch:
        result.append(_match_releases_by_tag_or_branch(
            repos_by_tag_or_branch, branches, default_branches, time_from, time_to, settings,
            mdb, pdb, cache))
    if repos_by_branch:
        result.append(_match_releases_by_branch(
            repos_by_branch, branches, default_branches, time_from, time_to, settings,
            mdb, pdb, cache))
        for k in repos_by_branch:
            applied_matches[k] = ReleaseMatch.branch
    result = await asyncio.gather(*result, return_exceptions=True)
    for i, r in enumerate(result):
        if isinstance(r, Exception):
            raise r
        if isinstance(r, tuple):
            r, am = r
            result[i] = r
            applied_matches.update(am)
    result = pd.concat(result) if result else dummy_releases_df()
    if index is not None:
        result.set_index(index, inplace=True)
    else:
        result.reset_index(drop=True, inplace=True)
    return result, applied_matches


def dummy_releases_df():
    """Create an empty releases DataFrame."""
    return pd.DataFrame(
        columns=[c.name for c in Release.__table__.columns] + [matched_by_column])


tag_by_branch_probe_lookaround = timedelta(weeks=4)


@sentry_span
async def _match_releases_by_tag_or_branch(repos: Iterable[str],
                                           branches: pd.DataFrame,
                                           default_branches: Dict[str, str],
                                           time_from: datetime,
                                           time_to: datetime,
                                           settings: Dict[str, ReleaseMatchSetting],
                                           mdb: databases.Database,
                                           pdb: databases.Database,
                                           cache: Optional[aiomcache.Client],
                                           ) -> Tuple[pd.DataFrame, Dict[str, ReleaseMatch]]:
    probe = await read_sql_query(
        select([distinct(Release.repository_full_name)])
        .where(and_(Release.repository_full_name.in_(repos),
                    Release.published_at.between(
            time_from - tag_by_branch_probe_lookaround,
            time_to + tag_by_branch_probe_lookaround),
        )),
        mdb, [Release.repository_full_name.key])
    matched = []
    repos_by_tag = probe[Release.repository_full_name.key].values
    if repos_by_tag.size > 0:
        matched.append(_match_releases_by_tag(repos_by_tag, time_from, time_to, settings, mdb))
    repos_by_branch = set(repos) - set(repos_by_tag)
    if repos_by_branch:
        matched.append(_match_releases_by_branch(
            repos_by_branch, branches, default_branches, time_from, time_to, settings,
            mdb, pdb, cache))
    matched = await asyncio.gather(*matched, return_exceptions=True)
    for m in matched:
        if isinstance(m, Exception):
            raise m
    applied_matches = {**{k: ReleaseMatch.tag for k in repos_by_tag},
                       **{k: ReleaseMatch.branch for k in repos_by_branch}}
    return pd.concat(matched), applied_matches


@sentry_span
async def _match_releases_by_tag(repos: Iterable[str],
                                 time_from: datetime,
                                 time_to: datetime,
                                 settings: Dict[str, ReleaseMatchSetting],
                                 db: databases.Database,
                                 ) -> pd.DataFrame:
    with sentry_sdk.start_span(op="fetch_tags"):
        releases = await read_sql_query(
            select([Release])
            .where(and_(Release.published_at.between(time_from, time_to),
                        Release.repository_full_name.in_(repos),
                        Release.commit_id.isnot(None)))
            .order_by(desc(Release.published_at)),
            db, Release, index=[Release.repository_full_name.key, Release.tag.key])
    releases = releases[~releases.index.duplicated(keep="first")]
    regexp_cache = {}
    matched = []
    prefix = PREFIXES["github"]
    for repo in repos:
        try:
            repo_releases = releases.loc[repo]
        except KeyError:
            continue
        if repo_releases.empty:
            continue
        regexp = settings[prefix + repo].tags
        if not regexp.endswith("$"):
            regexp += "$"
        # note: dict.setdefault() is not good here because re.compile() will be evaluated
        try:
            regexp = regexp_cache[regexp]
        except KeyError:
            regexp = regexp_cache[regexp] = re.compile(regexp)
        tags_matched = repo_releases.index[repo_releases.index.str.match(regexp)]
        matched.append([(repo, tag) for tag in tags_matched])
    # this shows up in the profile but I cannot make it faster
    releases = releases.loc[list(chain.from_iterable(matched))]
    releases.reset_index(inplace=True)
    releases[matched_by_column] = ReleaseMatch.tag.value
    return releases


@sentry_span
async def _match_releases_by_branch(repos: Iterable[str],
                                    branches: pd.DataFrame,
                                    default_branches: Dict[str, str],
                                    time_from: datetime,
                                    time_to: datetime,
                                    settings: Dict[str, ReleaseMatchSetting],
                                    mdb: databases.Database,
                                    pdb: databases.Database,
                                    cache: Optional[aiomcache.Client],
                                    ) -> pd.DataFrame:
    regexp_cache = {}
    branches_matched = {}
    prefix = PREFIXES["github"]
    branches = branches.take(np.where(branches[Branch.repository_full_name.key].isin(repos))[0])
    for repo, repo_branches in branches.groupby(Branch.repository_full_name.key, sort=False):
        regexp = settings[prefix + repo].branches
        default_branch = default_branches[repo]
        regexp = regexp.replace(default_branch_alias, default_branch)
        if not regexp.endswith("$"):
            regexp += "$"
        # note: dict.setdefault() is not good here because re.compile() will be evaluated
        try:
            regexp = regexp_cache[regexp]
        except KeyError:
            regexp = regexp_cache[regexp] = re.compile(regexp)
        matched = repo_branches[repo_branches[Branch.branch_name.key].str.match(regexp)]
        if not matched.empty:
            branches_matched[repo] = matched
    if not branches_matched:
        return dummy_releases_df()

    ghcfp = GitHubCommitFirstParents
    default_version = ghcfp.__table__.columns[ghcfp.format_version.key].default.arg

    async def _fetch_commit_first_parents_pdb():
        with sentry_sdk.start_span(op="_match_releases_by_branch/_fetch_commit_first_parents_pdb"):
            return await pdb.fetch_all(
                select([ghcfp.repository_full_name, ghcfp.commits])
                .where(and_(ghcfp.repository_full_name.in_(branches_matched),
                            ghcfp.format_version == default_version)))

    pre_mp_tasks = [
        _fetch_commit_first_parents_pdb(),
        _fetch_pr_merge_commits(branches_matched, time_from, time_to, mdb, cache),
    ]
    data_rows, pr_merge_commits = await asyncio.gather(*pre_mp_tasks, return_exceptions=True)
    for r in (data_rows, pr_merge_commits):
        if isinstance(r, Exception):
            raise r from None
    data_by_repo = {
        r[ghcfp.repository_full_name.key]: r[ghcfp.commits.key]
        for r in data_rows
    }
    del data_rows
    mp_tasks = [
        _fetch_merge_points(data_by_repo.get(repo), repo, branches[Branch.commit_id.key],
                            pr_merge_commits.get(repo), time_from, time_to, mdb, pdb, cache)
        for repo, branches in branches_matched.items()
    ]
    all_merge_commits = []
    with sentry_sdk.start_span(op="_match_releases_by_branch/_fetch_merge_points"):
        for mps in await asyncio.gather(*mp_tasks, return_exceptions=True):
            if isinstance(mps, Exception):
                raise mps from None
            all_merge_commits.extend(mps)
    all_commits = await _fetch_commits(all_merge_commits, mdb, cache)
    del all_merge_commits
    pseudo_releases = []
    for repo in branches_matched:
        commits = all_commits.take(
            np.where(all_commits[PushCommit.repository_full_name.key] == repo)[0])
        gh_merge = ((commits[PushCommit.committer_name.key] == "GitHub")
                    & (commits[PushCommit.committer_email.key] == "noreply@github.com"))
        commits[PushCommit.author_login.key].where(
            gh_merge, commits.loc[~gh_merge, PushCommit.committer_login.key], inplace=True)
        pseudo_releases.append(pd.DataFrame({
            Release.author.key: commits[PushCommit.author_login.key],
            Release.commit_id.key: commits[PushCommit.node_id.key],
            Release.id.key: commits[PushCommit.node_id.key],
            Release.name.key: commits[PushCommit.sha.key],
            Release.published_at.key: commits[PushCommit.committed_date.key],
            Release.repository_full_name.key: repo,
            Release.sha.key: commits[PushCommit.sha.key],
            Release.tag.key: None,
            Release.url.key: commits[PushCommit.url.key],
            matched_by_column: [ReleaseMatch.branch.value] * len(commits),
        }))
    if not pseudo_releases:
        return dummy_releases_df()
    pseudo_releases = pd.concat(pseudo_releases, copy=False)
    return pseudo_releases


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda branches_matched, time_from, time_to, **_: (
        ";".join("%s:%s" % (k, ",".join(sorted(v[Branch.branch_name.key].values)))
                 for k, v in branches_matched.items()),
        time_from.timestamp(), time_to.timestamp(),
    ),
    refresh_on_access=True,
)
async def _fetch_pr_merge_commits(branches_matched: Dict[str, pd.DataFrame],
                                  time_from: datetime,
                                  time_to: datetime,
                                  db: databases.Database,
                                  cache: Optional[aiomcache.Client]) -> Dict[str, List[str]]:
    branch_filters = []
    for repo, branches in branches_matched.items():
        branch_filters.append(and_(PullRequest.repository_full_name == repo,
                                   PullRequest.base_ref.in_(branches[Branch.branch_name.key])))
    rows = await db.fetch_all(
        select([PullRequest.repository_full_name, PullRequest.merge_commit_id])
        .where(and_(
            or_(*branch_filters),
            PullRequest.merged_at.between(time_from, time_to),
            PullRequest.merge_commit_id.isnot(None),
        )))
    pr_merge_commits_by_repo = {}
    for r in rows:
        pr_merge_commits_by_repo.setdefault(r[PullRequest.repository_full_name.key], []).append(
            r[PullRequest.merge_commit_id.key])
    return pr_merge_commits_by_repo


async def _fetch_merge_points(data: Optional[bytes],
                              repo: str,
                              commit_ids: Iterable[str],
                              merge_commit_ids: Optional[Iterable[str]],
                              time_from: datetime,
                              time_to: datetime,
                              mdb: databases.Database,
                              pdb: databases.Database,
                              cache: Optional[aiomcache.Client],
                              ) -> Set[str]:
    first_parents = await _fetch_first_parents(
        data, repo, commit_ids, time_from, time_to, mdb, pdb, cache)
    if merge_commit_ids is not None:
        first_parents.update(merge_commit_ids)
    return first_parents


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda commits, **_: (",".join(sorted(commits)),),
    refresh_on_access=True,
)
async def _fetch_commits(commits: List[str],
                         db: databases.Database,
                         cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    return await read_sql_query(
        select([PushCommit]).where(PushCommit.node_id.in_(commits))
        .order_by(desc(PushCommit.commit_date)),
        db, PushCommit)


@sentry_span
async def map_prs_to_releases(prs: pd.DataFrame,
                              releases: pd.DataFrame,
                              matched_bys: Dict[str, ReleaseMatch],
                              branches: pd.DataFrame,
                              default_branches: Dict[str, str],
                              time_to: datetime,
                              release_settings: Dict[str, ReleaseMatchSetting],
                              mdb: databases.Database,
                              pdb: databases.Database,
                              cache: Optional[aiomcache.Client],
                              ) -> pd.DataFrame:
    """Match the merged pull requests to the nearest releases that include them."""
    assert isinstance(time_to, datetime)
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    pr_releases = new_released_prs_df()
    if prs.empty:
        return pr_releases
    tasks = [
        discover_unreleased_prs(
            prs, releases, matched_bys, default_branches, release_settings, pdb),
        load_precomputed_pr_releases(
            prs.index, time_to, matched_bys, default_branches, release_settings, pdb, cache),
    ]
    unreleased_prs, precomputed_pr_releases = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (unreleased_prs, precomputed_pr_releases):
        if isinstance(r, Exception):
            raise r from None
    add_pdb_hits(pdb, "map_prs_to_releases/released", len(precomputed_pr_releases))
    add_pdb_hits(pdb, "map_prs_to_releases/unreleased", len(unreleased_prs))
    pr_releases = precomputed_pr_releases
    merged_prs = prs[~prs.index.isin(pr_releases.index.union(unreleased_prs))]
    tasks = [
        _map_prs_to_releases(merged_prs, releases, mdb, pdb, cache),
        _find_dead_merged_prs(merged_prs, branches, default_branches, mdb, pdb, cache),
        _fetch_labels(merged_prs.index, mdb),
    ]
    missed_released_prs, dead_prs, labels = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (missed_released_prs, dead_prs, labels):
        if isinstance(r, Exception):
            raise r from None
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
        merged_prs, missed_released_prs, releases, labels, matched_bys, default_branches,
        release_settings, pdb))
    return pr_releases.append(missed_released_prs)


async def _map_prs_to_releases(prs: pd.DataFrame,
                               releases: pd.DataFrame,
                               mdb: databases.Database,
                               pdb: databases.Database,
                               cache: Optional[aiomcache.Client],
                               ) -> pd.DataFrame:
    if prs.empty:
        return new_released_prs_df()
    releases = dict(list(releases.groupby(Release.repository_full_name.key, sort=False)))
    histories = await _fetch_release_histories(releases, mdb, pdb, cache)
    released_prs = []
    for repo, repo_prs in prs.groupby(PullRequest.repository_full_name.key, sort=False):
        try:
            repo_releases = releases[repo]
            history = histories[repo]
        except KeyError:
            # no releases exist for this repo
            continue
        for pr_id, merge_sha in zip(repo_prs.index,
                                    repo_prs[PullRequest.merge_commit_sha.key].values):
            try:
                items = history[merge_sha]
            except KeyError:
                continue
            ri = items[0]
            if ri < 0:
                continue
            r = repo_releases.xs(ri)
            released_prs.append((pr_id,
                                 r[Release.published_at.key],
                                 r[Release.author.key],
                                 r[Release.url.key],
                                 repo,
                                 r[matched_by_column]))
    released_prs = new_released_prs_df(released_prs)
    released_prs[Release.published_at.key] = np.maximum(
        released_prs[Release.published_at.key],
        prs.loc[released_prs.index, PullRequest.merged_at.key])
    return postprocess_datetime(released_prs)


@sentry_span
async def _find_dead_merged_prs(prs: pd.DataFrame,
                                branches: pd.DataFrame,
                                default_branches: Dict[str, str],
                                mdb: databases.Database,
                                pdb: databases.Database,
                                cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    if branches.empty:
        return new_released_prs_df()
    prs = prs.take(np.where(
        prs[PullRequest.merged_at.key] <= datetime.now(timezone.utc) - timedelta(hours=1))[0])
    # timedelta(hours=1) must match the `exptime` of `_fetch_repository_commits()`
    # commits DAGs are cached and may be not fully up to date, so otherwise some PRs may appear in
    # dead_prs and missed_released_prs at the same time
    # see also: DEV-554
    repos = prs[PullRequest.repository_full_name.key].unique()
    if len(repos) == 0:
        return new_released_prs_df()
    commits = await _fetch_repository_commits(repos, branches, default_branches, mdb, pdb, cache)
    rfnkey = PullRequest.repository_full_name.key
    mchkey = PullRequest.merge_commit_sha.key
    clskey = PullRequest.closed_at.key
    dead_prs = []
    for repo, repo_prs in prs[[mchkey, rfnkey, clskey]].groupby(rfnkey, sort=False):
        repo_commits = commits[repo]
        if len(repo_commits) == 0:
            # metadata branches fault
            continue
        repo_hashes = repo_prs[mchkey].values.astype("U40")
        indexes = np.searchsorted(repo_commits, repo_hashes)
        indexes[indexes == len(repo_commits)] = 0  # whatever index is fine
        dead_indexes = np.where(repo_hashes != repo_commits[indexes])[0]
        dead_prs.extend((pr_id, ct, None, None, repo, ReleaseMatch.force_push_drop)
                        for pr_id, ct in zip(repo_prs.index.values.take(dead_indexes),
                                             repo_prs[clskey].take(dead_indexes)))
    return new_released_prs_df(dead_prs)


@sentry_span
async def _fetch_labels(node_ids: Iterable[str], mdb: databases.Database) -> Dict[str, List[str]]:
    rows = await mdb.fetch_all(
        select([PullRequestLabel.pull_request_node_id, PullRequestLabel.name])
        .where(PullRequestLabel.pull_request_node_id.in_(node_ids)))
    labels = {}
    for row in rows:
        node_id, label = row[0], row[1]
        labels.setdefault(node_id, []).append(label)
    return labels


async def _fetch_release_histories(releases: Dict[str, pd.DataFrame],
                                   mdb: databases.Database,
                                   pdb: databases.Database,
                                   cache: Optional[aiomcache.Client],
                                   ) -> Dict[str, Dict[str, List[str]]]:
    log = logging.getLogger("%s._fetch_release_histories" % metadata.__package__)
    histories = {}
    pdags = await _fetch_precomputed_commit_histories(releases, pdb)

    async def fetch_release_history(repo, repo_releases):
        dag = await _fetch_commit_history_dag(
            pdags.get(repo), repo,
            repo_releases[Release.commit_id.key].values,
            repo_releases[Release.sha.key].values,
            repo_releases[Release.published_at.key].values,
            mdb, pdb, cache)
        histories[repo] = history = {k: [-1, *v] for k, v in dag.items()}
        release_hashes = set(repo_releases[Release.sha.key].values)
        for rel_index, rel_sha in zip(repo_releases.index.values,
                                      repo_releases[Release.sha.key].values):
            if rel_sha not in history:
                log.error("DEV-256 release commit %s was not found in the commit history",
                          rel_sha)
                continue
            update_history(history, rel_sha, rel_index, release_hashes)

    errors = await asyncio.gather(*(fetch_release_history(*r) for r in releases.items()),
                                  return_exceptions=True)
    for e in errors:
        if e is not None:
            raise e from None
    return histories


async def _fetch_precomputed_commit_histories(repos: Iterable[str],
                                              pdb: databases.Database) -> Dict[str, bytes]:
    default_version = GitHubCommitHistory.__table__ \
        .columns[GitHubCommitHistory.format_version.key].default.arg
    pdags = await pdb.fetch_all(
        select([GitHubCommitHistory.repository_full_name, GitHubCommitHistory.dag])
        .where(and_(GitHubCommitHistory.repository_full_name.in_(repos),
                    GitHubCommitHistory.format_version == default_version)))
    pdags = {r[GitHubCommitHistory.repository_full_name.key]: r[GitHubCommitHistory.dag.key]
             for r in pdags}
    return pdags


@cached(
    exptime=60 * 60,  # 1 hour
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda repo, commit_ids, time_from, time_to, **_: (
        ",".join(sorted(commit_ids)), time_from.timestamp(), time_to.timestamp(),
    ),
    refresh_on_access=True,
)
async def _fetch_first_parents(data: Optional[bytes],
                               repo: str,
                               commit_ids: Iterable[str],
                               time_from: datetime,
                               time_to: datetime,
                               mdb: databases.Database,
                               pdb: databases.Database,
                               cache: Optional[aiomcache.Client],
                               ) -> Set[str]:
    # Git parent-child is reversed github_node_commit_parents' parent-child.
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)

    if data is not None:
        f = BytesIO(data)
        first_parents = pickle.load(f)
        try:
            need_update = \
                (first_parents[np.searchsorted(first_parents, commit_ids)] != commit_ids).any()
        except IndexError:
            # np.searchsorted can return len(first_parents)
            need_update = True
        if not need_update:
            timestamps = pickle.load(f)
        else:
            del first_parents
        del f
        del data
    else:
        need_update = True

    if need_update:
        add_pdb_misses(pdb, "_fetch_first_parents", 1)
        quote = "`" if mdb.url.dialect == "sqlite" else ""
        query = f"""
            WITH RECURSIVE commit_first_parents AS (
                SELECT
                    p.child_id AS parent,
                    cc.id AS parent_id,
                    cc.committed_date as committed_date
                FROM
                    github_node_commit_parents p
                        LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                        LEFT JOIN github_node_commit cc ON p.child_id = cc.id
                WHERE
                        p.parent_id IN ('{"', '".join(commit_ids)}')
                    AND p.{quote}index{quote} = 0
                    AND cc.id IS NOT NULL
                UNION
                    SELECT
                        p.child_id AS parent,
                        cc.id AS parent_id,
                        cc.committed_date as committed_date
                    FROM
                        github_node_commit_parents p
                            INNER JOIN commit_first_parents h ON h.parent = p.parent_id
                            LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                            LEFT JOIN github_node_commit cc ON p.child_id = cc.id
                    WHERE p.{quote}index{quote} = 0 AND cc.id IS NOT NULL
            ) SELECT
                parent_id,
                committed_date
            FROM
                commit_first_parents
            UNION
                SELECT
                    id as parent_id,
                    committed_date
                FROM
                    github_node_commit
                WHERE
                    id IN ('{"', '".join(commit_ids)}');"""

        async with mdb.connection() as conn:
            if isinstance(conn.raw_connection, asyncpg.connection.Connection):
                # this works much faster then iterate() / fetch_all()
                async with conn._query_lock:
                    rows = await conn.raw_connection.fetch(query)
            else:
                rows = await conn.fetch_all(query)

        utc = timezone.utc
        first_parents = np.asarray([r[0] for r in rows])
        if mdb.url.dialect == "sqlite":
            timestamps = np.fromiter((str_to_datetime(r[1]).replace(tzinfo=utc) for r in rows),
                                     "datetime64[us]", count=len(rows))
        else:
            timestamps = np.fromiter((r[1] for r in rows), "datetime64[us]", count=len(rows))
        order = np.argsort(first_parents)
        first_parents = first_parents[order]
        timestamps = timestamps[order]
        f = BytesIO()
        pickle.dump(first_parents, f)
        pickle.dump(timestamps, f)
        values = GitHubCommitFirstParents(repository_full_name=repo, commits=f.getvalue()) \
            .create_defaults().explode(with_primary_keys=True)
        if pdb.url.dialect in ("postgres", "postgresql"):
            sql = postgres_insert(GitHubCommitFirstParents).values(values)
            sql = sql.on_conflict_do_update(
                constraint=GitHubCommitFirstParents.__table__.primary_key,
                set_={GitHubCommitFirstParents.commits.key: sql.excluded.commits,
                      GitHubCommitFirstParents.updated_at.key: sql.excluded.updated_at})
        elif pdb.url.dialect == "sqlite":
            sql = insert(GitHubCommitFirstParents).values(values).prefix_with("OR REPLACE")
        else:
            raise AssertionError("Unsupported database dialect: %s" % pdb.url.dialect)
        await defer(pdb.execute(sql))
    else:  # if need_update
        add_pdb_hits(pdb, "_fetch_first_parents", 1)

    time_from = time_from.replace(tzinfo=None)
    time_to = time_to.replace(tzinfo=None)
    result = set(first_parents[(time_from <= timestamps) & (timestamps < time_to)].tolist())
    return result


@cached(
    exptime=60 * 60,  # 1 hour
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda commit_ids, **_: (",".join(sorted(commit_ids)),),
    refresh_on_access=True,
)
async def _fetch_commit_history_dag(dag: Optional[bytes],
                                    repo: str,
                                    commit_ids: np.ndarray,
                                    commit_shas: np.ndarray,
                                    commit_dates: np.ndarray,
                                    mdb: databases.Database,
                                    pdb: databases.Database,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Dict[str, List[str]]:
    # Git parent-child is reversed github_node_commit_parents' parent-child.
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)

    if dag is not None:
        dag = marshal.loads(dag)
        need_update = False
        for commit_sha in commit_shas:
            if commit_sha not in dag:
                need_update = True
                break
        if not need_update:
            add_pdb_hits(pdb, "_fetch_commit_history_dag", 1)
            return dag

    add_pdb_misses(pdb, "_fetch_commit_history_dag", 1)
    threshold = 50
    if len(commit_ids) > threshold:
        order = np.argsort(commit_dates)
        _commit_ids = commit_ids[order]
        _commit_shas = commit_shas.astype("U40")[order]
    else:
        _commit_ids = commit_ids
        _commit_shas = commit_shas
    raw_dag = set()
    while len(_commit_ids) > threshold:
        rows = await _fetch_commit_history_edges([_commit_ids[-1]], mdb)
        if len(rows) > 0:
            raw_dag.update(rows)
            left_mask = ~np.isin(_commit_shas, np.fromiter((r[1] for r in rows), "U40", len(rows)))
            _commit_shas = _commit_shas[left_mask]
            _commit_ids = _commit_ids[left_mask]
        else:
            _commit_ids = _commit_ids[:-1]
            _commit_shas = _commit_shas[:-1]
    if len(_commit_ids) > 0:
        raw_dag.update(await _fetch_commit_history_edges(_commit_ids, mdb))
    dag = {}
    for child, parent in raw_dag:
        # reverse the order so that parent-child matches github_node_commit_parents again
        try:
            dag[parent].append(child)
        except KeyError:
            # first iteration
            dag[parent] = [child]
        dag.setdefault(child, [])
    if not dag:
        # initial commit(s)
        return {sha: [] for sha in commit_shas}
    else:
        values = GitHubCommitHistory(repository_full_name=repo, dag=marshal.dumps(dag)) \
            .create_defaults().explode(with_primary_keys=True)
        if pdb.url.dialect in ("postgres", "postgresql"):
            sql = postgres_insert(GitHubCommitHistory).values(values)
            sql = sql.on_conflict_do_update(
                constraint=GitHubCommitHistory.__table__.primary_key,
                set_={GitHubCommitHistory.dag.key: sql.excluded.dag,
                      GitHubCommitHistory.updated_at.key: sql.excluded.updated_at})
        elif pdb.url.dialect == "sqlite":
            sql = insert(GitHubCommitHistory).values(values).prefix_with("OR REPLACE")
        else:
            raise AssertionError("Unsupported database dialect: %s" % pdb.url.dialect)
        await defer(pdb.execute(sql))
    return dag


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, branches, **_: (
        ",".join(sorted(repos)), ",".join(np.sort(branches[Branch.commit_sha.key].values))),
    refresh_on_access=True,
)
async def _fetch_repository_commits(repos: Collection[str],
                                    branches: pd.DataFrame,
                                    default_branches: Dict[str, str],
                                    mdb: databases.Database,
                                    pdb: databases.Database,
                                    cache: Optional[aiomcache.Client]) -> Dict[str, np.ndarray]:
    filters = []
    branch_names = branches[Branch.branch_name.key].values
    branch_commits = branches[Branch.commit_sha.key].values
    branch_repos = branches[Branch.repository_full_name.key].values
    sqlite = pdb.url.dialect == "sqlite"
    ghrc = GitHubRepositoryCommits
    for repo in repos:
        matched_rows = branch_repos == repo
        pairs = zip(branch_names[matched_rows], branch_commits[matched_rows])
        if sqlite:
            pairs = sorted(pairs)
        heads = dict(pairs)
        filters.append(and_(ghrc.repository_full_name == repo, ghrc.heads == heads))
    with sentry_sdk.start_span(op="_fetch_repository_commits/pdb"):
        rows = await pdb.fetch_all(
            select([ghrc.repository_full_name, ghrc.hashes])
            .where(and_(
                ghrc.format_version == ghrc.__table__.columns[ghrc.format_version.key].default.arg,
                or_(*filters),
            )))
    result = {row[0]: pickle.loads(row[1]) for row in rows}
    add_pdb_hits(pdb, "_fetch_repository_commits", len(result))
    add_pdb_misses(pdb, "_fetch_repository_commits", len(repos) - len(result))
    missed_repos = [repo for repo in repos if repo not in result]
    branches = branches.take(np.where(np.in1d(branch_repos, missed_repos))[0])
    branches.sort_values(
        [Branch.commit_date.key], ascending=False, ignore_index=True, inplace=True)
    grouped_branches = list(branches.groupby(Branch.repository_full_name.key, sort=False))
    # fetch the default branch first
    tasks = []
    for repo, branches in grouped_branches:
        try:
            commit_id = branches[Branch.commit_id.key].iloc[
                np.where(branches[Branch.branch_name.key] == default_branches[repo])[0][0]]
        except IndexError:
            # there is a metadata problem and we could not find the default branch
            continue
        tasks.append(_fetch_commit_history_hashes([commit_id], mdb))
    with sentry_sdk.start_span(op="_fetch_repository_commits/_fetch_hashes/default"):
        default_hashes = await asyncio.gather(*tasks, return_exceptions=True)
    for r in default_hashes:
        if isinstance(r, Exception):
            raise r from None
    # fetch the rest of the branches
    tasks = []
    for (repo, branches), all_hashes in zip(grouped_branches, default_hashes):
        missing_hashes = set(branches[Branch.commit_sha.key].values) - all_hashes
        missing_mask = branches[Branch.commit_sha.key].isin(missing_hashes)
        commit_ids = branches[Branch.commit_id.key].values[missing_mask]
        commit_hashes = branches[Branch.commit_sha.key].values[missing_mask]
        heads = zip(branches[Branch.branch_name.key].values,
                    branches[Branch.commit_sha.key].values)
        if sqlite:
            heads = sorted(heads)
        tasks.append(_fetch_commit_history_hashes_batched(
            all_hashes, commit_ids, commit_hashes, repo, dict(heads), mdb, pdb))
    with sentry_sdk.start_span(op="_fetch_repository_commits/_fetch_hashes/branches"):
        missed = await asyncio.gather(*tasks, return_exceptions=True)
    for (repo, _), arr in zip(grouped_branches, missed):
        if isinstance(arr, Exception):
            raise arr from None
        result[repo] = arr
    # some repos may have 0 branches due to a metadata fault
    log = logging.getLogger("%s._fetch_repository_commits" % metadata.__package__)
    for repo in repos:
        if repo not in result:
            result[repo] = np.empty((0,), dtype="U40")
            log.warning("No branches in %s", repo)
    return result


async def _fetch_commit_history_hashes_batched(hashes: Set[str],
                                               commit_ids: Sequence[str],
                                               commit_hashes: Sequence[str],
                                               repo: str,
                                               heads: Dict[str, str],
                                               mdb: databases.Database,
                                               pdb: databases.Database) -> np.ndarray:
    batch_size = 20
    while len(commit_ids) > 0:
        hashes = hashes.union(await _fetch_commit_history_hashes(commit_ids[:batch_size], mdb))
        new_commit_ids = []
        new_commit_hashes = []
        for cid, chash in zip(commit_ids[batch_size:], commit_hashes[batch_size:]):
            if chash not in hashes:
                new_commit_ids.append(cid)
                new_commit_hashes.append(chash)
        commit_ids, commit_hashes = new_commit_ids, new_commit_hashes
    hashes = np.sort(np.fromiter(hashes, count=len(hashes), dtype="U40"))
    values = GitHubRepositoryCommits(
        repository_full_name=repo, heads=heads, hashes=pickle.dumps(hashes),
    ).create_defaults().explode(with_primary_keys=True)
    if pdb.url.dialect in ("postgres", "postgresql"):
        sql = postgres_insert(GitHubRepositoryCommits).values(values)
        sql = sql.on_conflict_do_update(
            constraint=GitHubRepositoryCommits.__table__.primary_key,
            set_={GitHubRepositoryCommits.hashes.key: sql.excluded.hashes,
                  GitHubRepositoryCommits.heads.key: sql.excluded.heads,
                  GitHubRepositoryCommits.updated_at.key: sql.excluded.updated_at})
    elif pdb.url.dialect == "sqlite":
        sql = insert(GitHubRepositoryCommits).values(values).prefix_with("OR REPLACE")
    else:
        raise AssertionError("Unsupported database dialect: %s" % pdb.url.dialect)
    await defer(pdb.execute(sql))
    return hashes


async def _fetch_commit_history_edges(commit_ids: Iterable[str],
                                      mdb: databases.Database) -> List[Tuple]:
    # query credits: @dennwc
    query = f"""
            WITH RECURSIVE commit_history AS (
                SELECT
                    p.child_id AS parent,
                    pc.oid AS child_oid,
                    cc.oid AS parent_oid
                FROM
                    github_node_commit_parents p
                        LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                        LEFT JOIN github_node_commit cc ON p.child_id = cc.id
                WHERE
                    p.parent_id IN ('{"', '".join(commit_ids)}')
                UNION
                    SELECT
                        p.child_id AS parent,
                        pc.oid AS child_oid,
                        cc.oid AS parent_oid
                    FROM
                        github_node_commit_parents p
                            INNER JOIN commit_history h ON h.parent = p.parent_id
                            LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                            LEFT JOIN github_node_commit cc ON p.child_id = cc.id
            ) SELECT
                parent_oid,
                child_oid
            FROM
                commit_history;"""
    async with mdb.connection() as conn:
        if isinstance(conn.raw_connection, asyncpg.connection.Connection):
            # this works much faster then iterate() / fetch_all()
            async with conn._query_lock:
                return await conn.raw_connection.fetch(query)
        else:
            return [tuple(r) for r in await conn.fetch_all(query)]


async def _fetch_commit_history_hashes(commit_ids: Iterable[str],
                                       mdb: databases.Database) -> Set[str]:
    query = f"""
        WITH RECURSIVE commit_history AS (
            SELECT
                p.child_id AS parent,
                pc.oid AS child_oid
            FROM
                github_node_commit_parents p
                    LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                    LEFT JOIN github_node_commit cc ON p.child_id = cc.id
            WHERE
                p.parent_id IN ('{"', '".join(commit_ids)}')
            UNION
                SELECT
                    p.child_id AS parent,
                    pc.oid AS child_oid
                FROM
                    github_node_commit_parents p
                        INNER JOIN commit_history h ON h.parent = p.parent_id
                        LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                        LEFT JOIN github_node_commit cc ON p.child_id = cc.id
        ) SELECT
            child_oid
        FROM
            commit_history
        UNION
            SELECT
                oid as child_id
            FROM
                github_node_commit
            WHERE
                id IN ('{"', '".join(commit_ids)}');"""
    async with mdb.connection() as conn:
        if isinstance(conn.raw_connection, asyncpg.connection.Connection):
            # this works much faster then iterate() / fetch_all()
            async with conn._query_lock:
                return {r[0] for r in await conn.raw_connection.fetch(query)}
        else:
            return {r[0] for r in await conn.fetch_all(query)}


async def _find_old_released_prs(releases: pd.DataFrame,
                                 pdag: Optional[bytes],
                                 time_boundary: datetime,
                                 authors: Collection[str],
                                 mergers: Collection[str],
                                 pr_blacklist: Optional[BinaryExpression],
                                 mdb: databases.Database,
                                 pdb: databases.Database,
                                 cache: Optional[aiomcache.Client],
                                 ) -> Iterable[Mapping]:
    observed_commits, _, _ = await _extract_released_commits(
        releases, pdag, time_boundary, mdb, pdb, cache)
    repo = releases.iloc[0][Release.repository_full_name.key] if not releases.empty else ""
    filters = [
        PullRequest.merged_at < time_boundary,
        PullRequest.repository_full_name == repo,
        PullRequest.merge_commit_sha.in_(observed_commits),
        PullRequest.hidden.is_(False),
    ]
    if len(authors) and len(mergers):
        filters.append(or_(
            PullRequest.user_login.in_(authors),
            PullRequest.merged_by_login.in_(mergers),
        ))
    elif len(authors):
        filters.append(PullRequest.user_login.in_(authors))
    elif len(mergers):
        filters.append(PullRequest.merged_by_login.in_(mergers))
    if pr_blacklist is not None:
        filters.append(pr_blacklist)
    with sentry_sdk.start_span(op="_find_old_released_prs/mdb"):
        return await mdb.fetch_all(select([PullRequest]).where(and_(*filters)))


@sentry_span
async def _extract_released_commits(releases: pd.DataFrame,
                                    pdag: Optional[bytes],
                                    time_boundary: datetime,
                                    mdb: databases.Database,
                                    pdb: databases.Database,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Tuple[Dict[str, List[str]], pd.DataFrame, Dict[str, str]]:
    log = logging.getLogger("%s._extract_released_commits" % metadata.__package__)
    repo = releases[Release.repository_full_name.key].unique()
    assert len(repo) == 1
    repo = repo[0]
    resolved_releases = set()
    hash_to_release = {h: rid for rid, h in zip(releases.index, releases[Release.sha.key].values)}
    time_mask = releases[Release.published_at.key] >= time_boundary
    new_releases = releases.take(np.where(time_mask)[0])
    assert not new_releases.empty, "you must check this before calling me"
    # we stop walking the DAG when we encounter these
    boundary_releases = set(releases.index.take(np.where(~time_mask)[0]))
    # original DAG with all the mentioned releases
    dag = await _fetch_commit_history_dag(
        pdag, repo,
        releases[Release.commit_id.key].values,
        releases[Release.sha.key].values,
        releases[Release.published_at.key].values,
        mdb, pdb, cache)

    for rid, root in zip(new_releases.index, new_releases[Release.sha.key].values):
        if rid in resolved_releases:
            continue
        parents = [root]
        visited = set()
        while parents:
            x = parents.pop()
            if x in visited:
                continue
            else:
                visited.add(x)
            xrid = hash_to_release.get(x)
            if xrid is not None:
                pubdt = releases.loc[xrid, Release.published_at.key]
                if pubdt >= time_boundary:
                    resolved_releases.add(xrid)
                else:
                    continue
            try:
                parents.extend(dag[x])
            except KeyError:
                log.error("DEV-256 missing commit parent for %s", x)

    # we need to traverse full history from boundary_releases and subtract it from the full DAG
    ignored_commits = set()
    for rid in boundary_releases:
        release_sha = releases.loc[rid, Release.sha.key]
        if release_sha in ignored_commits:
            continue
        parents = [release_sha]
        while parents:
            x = parents.pop()
            if x in ignored_commits:
                continue
            ignored_commits.add(x)
            children = dag[x]
            parents.extend(children)
    for c in ignored_commits:
        try:
            del dag[c]
        except KeyError:
            # may raise on boundary release commits
            continue
    return dag, new_releases, hash_to_release


@sentry_span
async def map_releases_to_prs(repos: Iterable[str],
                              branches: pd.DataFrame,
                              default_branches: Dict[str, str],
                              time_from: datetime,
                              time_to: datetime,
                              authors: Collection[str],
                              mergers: Collection[str],
                              release_settings: Dict[str, ReleaseMatchSetting],
                              mdb: databases.Database,
                              pdb: databases.Database,
                              cache: Optional[aiomcache.Client],
                              pr_blacklist: Optional[BinaryExpression] = None,
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, ReleaseMatch]]:
    """Find pull requests which were released between `time_from` and `time_to` but merged before \
    `time_from`.

    :param authors: Required PR authors.
    :param mergers: Required PR mergers.
    :return: pd.DataFrame with found PRs that were created before `time_from` and released \
             between `time_from` and `time_to` \
             + \
             pd.DataFrame with the discovered releases between `time_from` and `time_to` \
             + \
             `matched_bys` so that we don't have to compute that mapping again.
    """
    assert isinstance(time_from, datetime)
    assert isinstance(time_to, datetime)
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)

    matched_bys, pdags, releases, releases_new = await _find_releases_for_matching_prs(
        repos, time_from, time_to, branches, default_branches, release_settings, pdb, mdb, cache)
    prs = []
    for repo, repo_releases in releases.groupby(Release.repository_full_name.key, sort=False):
        if (repo_releases[Release.published_at.key] >= time_from).any():
            prs.append(_find_old_released_prs(
                repo_releases, pdags.get(repo), time_from, authors, mergers, pr_blacklist,
                mdb, pdb, cache))
    if prs:
        with sentry_sdk.start_span(op="_find_old_released_prs"):
            prs = await asyncio.gather(*prs, return_exceptions=True)
        for pr in prs:
            if isinstance(pr, Exception):
                raise pr
        return (
            wrap_sql_query(chain.from_iterable(prs), PullRequest, index=PullRequest.node_id.key),
            releases_new,
            matched_bys,
        )
    return (
        pd.DataFrame(columns=[c.name for c in PullRequest.__table__.columns
                              if c.name != PullRequest.node_id.key]),
        releases_new,
        matched_bys,
    )


@sentry_span
async def _find_releases_for_matching_prs(repos, time_from, time_to, branches, default_branches,
                                          release_settings, pdb, mdb, cache):
    # we have to load releases in two separate batches: before and after time_from
    # that's because the release strategy can change depending on the time range
    # see ENG-710 and ENG-725
    tasks = [
        load_releases(repos, branches, default_branches, time_from, time_to, release_settings,
                      mdb, pdb, cache),
        _fetch_precomputed_commit_histories(repos, pdb),
    ]
    releases_new, pdags = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (releases_new, pdags):
        if isinstance(r, Exception):
            raise r from None
    releases_new, matched_bys = releases_new
    # these matching rules must be applied in the past to stay consistent
    prefix = PREFIXES["github"]
    consistent_release_settings = {}
    for repo in matched_bys:
        setting = release_settings[prefix + repo]
        consistent_release_settings[prefix + repo] = ReleaseMatchSetting(
            tags=setting.tags,
            branches=setting.branches,
            match=ReleaseMatch(matched_bys[repo]),
        )
    # let's try to find the releases not older than 5 weeks before `time_from`
    lookbehind_time_from = time_from - timedelta(days=5 * 7)
    tasks = [
        _fetch_repository_first_commit_dates(repos, mdb, pdb),
        load_releases(matched_bys, branches, default_branches, lookbehind_time_from, time_from,
                      consistent_release_settings, mdb, pdb, cache),
    ]
    repo_births, releases_old = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (repo_births, releases_old):
        if isinstance(r, Exception):
            raise r from None
    releases_old = releases_old[0]
    hard_repos = set(matched_bys) - set(releases_old[Release.repository_full_name.key].unique())
    if hard_repos:
        with sentry_sdk.start_span(op="_find_releases_for_matching_prs/hard_repos"):
            repo_births = sorted(
                (row["min"], row[PushCommit.repository_full_name.key])
                for row in repo_births
                if row[PushCommit.repository_full_name.key] in hard_repos
            )
            repo_births_dates = [rb[0].replace(tzinfo=timezone.utc) for rb in repo_births]
            repo_births_names = [rb[1] for rb in repo_births]
            del repo_births
            deeper_step = timedelta(days=6 * 31)
            while hard_repos:
                # no previous releases were discovered for `hard_repos`, go deeper in history
                hard_repos = hard_repos.intersection(repo_births_names[:bisect.bisect_right(
                    repo_births_dates, lookbehind_time_from)])
                if not hard_repos:
                    break
                releases_old_hard, _ = await load_releases(
                    hard_repos, branches, default_branches, lookbehind_time_from - deeper_step,
                    lookbehind_time_from, consistent_release_settings, mdb, pdb, cache)
                releases_old = releases_old.append(releases_old_hard)
                hard_repos -= set(releases_old_hard[Release.repository_full_name.key].unique())
                del releases_old_hard
                lookbehind_time_from -= deeper_step
    releases = releases_new.append(releases_old)
    releases.reset_index(drop=True, inplace=True)
    return matched_bys, pdags, releases, releases_new


@sentry_span
async def _fetch_repository_first_commit_dates(repos: Iterable[str],
                                               mdb: databases.Database,
                                               pdb: databases.Database,
                                               ) -> List[Mapping]:
    result = await pdb.fetch_all(
        select([GitHubRepository.repository_full_name,
                GitHubRepository.first_commit.label("min")])
        .where(GitHubRepository.repository_full_name.in_(repos)))
    add_pdb_hits(pdb, "_fetch_repository_first_commit_dates", len(result))
    missing = set(repos) - {r[0] for r in result}
    add_pdb_misses(pdb, "_fetch_repository_first_commit_dates", len(missing))
    if missing:
        computed = await mdb.fetch_all(
            select([NodeRepository.name_with_owner.label(PushCommit.repository_full_name.key),
                    func.min(NodeCommit.committed_date).label("min"),
                    NodeRepository.id])
            .select_from(join(NodeCommit, NodeRepository,
                              NodeCommit.repository == NodeRepository.id))
            .where(NodeRepository.name_with_owner.in_(missing))
            .group_by(NodeRepository.id))
        if computed:
            values = [GitHubRepository(repository_full_name=r[0], first_commit=r[1], node_id=r[2])
                      .create_defaults().explode(with_primary_keys=True)
                      for r in computed]

            async def insert_repository():
                try:
                    await pdb.execute_many(insert(GitHubRepository), values)
                except Exception as e:
                    log = logging.getLogger(
                        "%s._fetch_repository_first_commit_dates" % metadata.__package__)
                    log.warning("Failed to store %d rows: %s: %s",
                                len(values), type(e).__name__, e)
            await defer(insert_repository())
            result.extend(computed)
    return result


@sentry_span
async def mine_releases(releases: pd.DataFrame,
                        time_boundary: datetime,
                        mdb: databases.Database,
                        pdb: databases.Database,
                        cache: Optional[aiomcache.Client]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collect details about each release published after `time_boundary` and calculate added \
    and deleted line statistics."""
    assert isinstance(mdb, databases.Database)
    assert isinstance(pdb, databases.Database)
    pdags = await _fetch_precomputed_commit_histories(
        releases[Release.repository_full_name.key].unique(), pdb)
    miners = (
        _mine_monorepo_releases(repo_releases, pdags.get(repo), time_boundary, mdb, pdb, cache)
        for repo, repo_releases in releases.groupby(Release.repository_full_name.key, sort=False)
        if (repo_releases[Release.published_at.key] >= time_boundary).any()
    )
    stats = await asyncio.gather(*miners, return_exceptions=True)
    for s in stats:
        if isinstance(s, BaseException):
            raise s from None
    user_columns = [User.login, User.avatar_url]
    if stats:
        stats = pd.concat(stats, copy=False)
        people = set(chain(chain.from_iterable(stats["commit_authors"]), stats["publisher"]))
        prefix = PREFIXES["github"]
        stats["publisher"] = prefix + stats["publisher"]
        stats["repository"] = prefix + stats["repository"]
        for calist in stats["commit_authors"].values:
            for i, v in enumerate(calist):
                calist[i] = prefix + v
        avatars = await read_sql_query(
            select(user_columns).where(User.login.in_(people)), mdb, user_columns)
        avatars[User.login.key] = prefix + avatars[User.login.key]
        return stats, avatars
    return pd.DataFrame(), pd.DataFrame(columns=[c.key for c in user_columns])


@cached(
    exptime=10 * 60,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda releases, **_: (sorted(releases.index),),
    refresh_on_access=True,
)
async def _mine_monorepo_releases(releases: pd.DataFrame,
                                  pdag: Optional[bytes],
                                  time_boundary: datetime,
                                  mdb: databases.Database,
                                  pdb: databases.Database,
                                  cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    dag, new_releases, hash_to_release = await _extract_released_commits(
        releases, pdag, time_boundary, mdb, pdb, cache)
    stop_hashes = set(new_releases[Release.sha.key])
    owned_commits = {}  # type: Dict[str, Set[str]]
    neighbours = {}  # type: Dict[str, Set[str]]

    def find_owned_commits(sha):
        try:
            return owned_commits[sha]
        except KeyError:
            accessible, boundaries, leaves = _traverse_commits(dag, sha, stop_hashes)
            neighbours[sha] = boundaries.union(leaves)
            for b in boundaries:
                accessible -= find_owned_commits(b)
            owned_commits[sha] = accessible
            return accessible

    for sha in new_releases[Release.sha.key].values:
        find_owned_commits(sha)
    data = []
    commit_df_columns = [PushCommit.additions, PushCommit.deletions, PushCommit.author_login]
    for release in new_releases.itertuples():
        sha = getattr(release, Release.sha.key)
        included_commits = owned_commits[sha]
        repo = getattr(release, Release.repository_full_name.key)
        df = await read_sql_query(
            select(commit_df_columns)
            .where(and_(PushCommit.repository_full_name == repo,
                        PushCommit.sha.in_(included_commits))),
            mdb, commit_df_columns)
        try:
            previous_published_at = max(releases.loc[hash_to_release[n], Release.published_at.key]
                                        for n in neighbours[sha] if n in hash_to_release)
        except ValueError:
            # no previous releases
            previous_published_at = await mdb.fetch_val(
                select([func.min(PushCommit.committed_date)])
                .where(and_(PushCommit.repository_full_name.in_(repo),
                            PushCommit.sha.in_(included_commits))))
        published_at = getattr(release, Release.published_at.key)
        data.append([
            getattr(release, Release.name.key) or getattr(release, Release.tag.key),
            repo,
            getattr(release, Release.url.key),
            published_at,
            (published_at - previous_published_at) if previous_published_at is not None
            else timedelta(0),
            df[PushCommit.additions.key].sum(),
            df[PushCommit.deletions.key].sum(),
            len(included_commits),
            getattr(release, Release.author.key),
            sorted(set(df[PushCommit.author_login.key]) - {None}),
        ])
    return pd.DataFrame.from_records(data, columns=[
        "name", "repository", "url", "published", "age", "added_lines", "deleted_lines", "commits",
        "publisher", "commit_authors"])


def _traverse_commits(dag: Dict[str, List[str]],
                      root: str,
                      stops: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    parents = [root]
    visited = set()
    boundaries = set()
    leaves = set()
    while parents:
        x = parents.pop()
        if x in visited:
            continue
        if x in stops and x != root:
            boundaries.add(x)
            continue
        try:
            children = dag[x]
            parents.extend(children)
        except KeyError:
            leaves.add(x)
            continue
        visited.add(x)
    return visited, boundaries, leaves
