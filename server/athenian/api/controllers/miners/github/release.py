import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain, groupby
import logging
import marshal
import pickle
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import aiomcache
import aiosqlite
import asyncpg
import databases
import numpy as np
import pandas as pd
from sqlalchemy import and_, desc, distinct, func, select

from athenian.api import metadata
from athenian.api.async_read_sql_query import postprocess_datetime, read_sql_query
from athenian.api.cache import cached, gen_cache_key, max_exptime
from athenian.api.controllers.miners.github.release_accelerated import update_history
from athenian.api.controllers.settings import default_branch_alias, Match, ReleaseMatchSetting
from athenian.api.models.metadata.github import Branch, PullRequest, PushCommit, Release, User
from athenian.api.typing_utils import DatabaseLike


matched_by_column = "matched_by"


async def load_releases(repos: Iterable[str],
                        time_from: datetime,
                        time_to: datetime,
                        settings: Dict[str, ReleaseMatchSetting],
                        conn: databases.core.Connection,
                        cache: Optional[aiomcache.Client],
                        index: Optional[Union[str, Sequence[str]]] = None,
                        ) -> pd.DataFrame:
    """
    Fetch releases from the metadata DB according to the match settings.

    :param repos: Repositories in which to search for releases *without the service prefix*.
    """
    assert isinstance(conn, databases.core.Connection)
    repos_by_tag_only = []
    repos_by_tag_or_branch = []
    repos_by_branch = []
    for repo in repos:
        v = settings["github.com/" + repo]
        if v.match == Match.tag:
            repos_by_tag_only.append(repo)
        elif v.match == Match.tag_or_branch:
            repos_by_tag_or_branch.append(repo)
        elif v.match == Match.branch:
            repos_by_branch.append(repo)
    result = []
    if repos_by_tag_only:
        result.append(await _match_releases_by_tag(
            repos_by_tag_only, time_from, time_to, settings, conn))
    if repos_by_tag_or_branch:
        result.append(await _match_releases_by_tag_or_branch(
            repos_by_tag_or_branch, time_from, time_to, settings, conn, cache))
    if repos_by_branch:
        result.append(await _match_releases_by_branch(
            repos_by_branch, time_from, time_to, settings, conn, cache))
    result = pd.concat(result) if result else pd.DataFrame()
    if len(result.columns) == 0:
        # can happen if no such repositories were found
        result = pd.DataFrame(columns=[c.name for c in Release.__table__.columns])
    if index is not None:
        result.set_index(index, inplace=True)
    return result


tag_by_branch_probe_lookaround = timedelta(weeks=4)


async def _match_releases_by_tag_or_branch(repos: Iterable[str],
                                           time_from: datetime,
                                           time_to: datetime,
                                           settings: Dict[str, ReleaseMatchSetting],
                                           conn: databases.core.Connection,
                                           cache: Optional[aiomcache.Client],
                                           ) -> pd.DataFrame:
    probe = await read_sql_query(
        select([distinct(Release.repository_full_name)])
        .where(and_(Release.repository_full_name.in_(repos),
                    Release.published_at.between(
            time_from - tag_by_branch_probe_lookaround,
            time_to + tag_by_branch_probe_lookaround),
        )),
        conn, [Release.repository_full_name.key])
    matched = []
    repos_by_tag = probe[Release.repository_full_name.key].values
    if repos_by_tag.size > 0:
        matched.append(await _match_releases_by_tag(
            repos_by_tag, time_from, time_to, settings, conn))
    repos_by_branch = set(repos) - set(repos_by_tag)
    if repos_by_branch:
        matched.append(await _match_releases_by_branch(
            repos_by_branch, time_from, time_to, settings, conn, cache))
    return pd.concat(matched)


async def _match_releases_by_tag(repos: Iterable[str],
                                 time_from: datetime,
                                 time_to: datetime,
                                 settings: Dict[str, ReleaseMatchSetting],
                                 conn: databases.core.Connection,
                                 ) -> pd.DataFrame:
    releases = await read_sql_query(
        select([Release])
        .where(and_(Release.published_at.between(time_from, time_to),
                    Release.repository_full_name.in_(repos)))
        .order_by(desc(Release.published_at)),
        conn, Release, index=[Release.repository_full_name.key, Release.tag.key])
    releases = releases[~releases.index.duplicated(keep="first")]
    regexp_cache = {}
    matched = []
    for repo in repos:
        try:
            repo_releases = releases.loc[repo]
        except KeyError:
            continue
        if repo_releases.empty:
            continue
        regexp = settings["github.com/" + repo].tags
        # note: dict.setdefault() is not good here because re.compile() will be evaluated
        try:
            regexp = regexp_cache[regexp]
        except KeyError:
            regexp = regexp_cache[regexp] = re.compile(regexp)
        tags_matched = repo_releases.index[repo_releases.index.str.match(regexp)]
        matched.append([(repo, tag) for tag in tags_matched])
    releases = releases.loc[list(chain.from_iterable(matched))]
    releases.reset_index(inplace=True)
    releases[matched_by_column] = Match.tag.value
    return releases


async def _match_releases_by_branch(repos: Iterable[str],
                                    time_from: datetime,
                                    time_to: datetime,
                                    settings: Dict[str, ReleaseMatchSetting],
                                    conn: databases.core.Connection,
                                    cache: Optional[aiomcache.Client],
                                    ) -> pd.DataFrame:
    branches = await read_sql_query(
        select([Branch]).where(Branch.repository_full_name.in_(repos)), conn, Branch)
    regexp_cache = {}
    branches_matched = []
    log = logging.getLogger("%s.match_releases_by_branch" % metadata.__package__)
    for repo, repo_branches in branches.groupby(Branch.repository_full_name.key):
        regexp = settings["github.com/" + repo].branches
        try:
            default_branch = \
                repo_branches[Branch.branch_name.key][repo_branches[Branch.is_default.key]].iloc[0]
        except IndexError:
            log.error('failed to find the default branch for "%s": only have %s',
                      repo, repo_branches[[Branch.branch_name.key, Branch.is_default.key]])
            continue
        regexp = regexp.replace(default_branch_alias, default_branch)
        # note: dict.setdefault() is not good here because re.compile() will be evaluated
        try:
            regexp = regexp_cache[regexp]
        except KeyError:
            regexp = regexp_cache[regexp] = re.compile(regexp)
        branches_matched.append(
            repo_branches[repo_branches[Branch.branch_name.key].str.match(regexp)])
    if not branches_matched:
        return pd.DataFrame()
    branches_matched = pd.concat(branches_matched)
    merge_points_by_repo = defaultdict(set)
    for repo, commit_id in zip(branches_matched[Branch.repository_full_name.key].values,
                               branches_matched[Branch.commit_id.key].values):
        merge_points_by_repo[repo].update(await _fetch_first_parents(commit_id, conn, cache))
    pseudo_releases = []

    async def fetch_merge_commits(repo, merge_points):
        commits = await read_sql_query(
            select([PushCommit]).where(and_(
                PushCommit.node_id.in_(merge_points),
                PushCommit.committed_date.between(time_from, time_to)))
            .order_by(desc(PushCommit.commit_date)),
            conn, PushCommit)
        gh_merge = ((commits[PushCommit.committer_name.key] == "GitHub")
                    & (commits[PushCommit.committer_email.key] == "noreply@github.com"))
        commits[PushCommit.author_login.key].where(
            gh_merge, commits.loc[~gh_merge, PushCommit.committer_login.key], inplace=True)
        pseudo_releases.append(pd.DataFrame({
            Release.author.key: commits[PushCommit.author_login.key],
            Release.commit_id.key: commits[PushCommit.node_id.key],
            Release.id.key: commits[PushCommit.node_id.key] + "_" + repo,
            Release.name.key: commits[PushCommit.sha.key],
            Release.published_at.key: commits[PushCommit.committed_date.key],
            Release.repository_full_name.key: repo,
            Release.sha.key: commits[PushCommit.sha.key],
            Release.tag.key: None,
            Release.url.key: commits[PushCommit.url.key],
            matched_by_column: [Match.branch.value] * len(commits),
        }))
    errors = await asyncio.gather(
        *(fetch_merge_commits(*rm) for rm in merge_points_by_repo.items()), return_exceptions=True)
    for e in errors:
        if e is not None:
            raise e from None
    return pd.concat(pseudo_releases)


async def map_prs_to_releases(prs: pd.DataFrame,
                              time_from: datetime,
                              time_to: datetime,
                              release_settings: Dict[str, ReleaseMatchSetting],
                              db: databases.Database,
                              cache: Optional[aiomcache.Client],
                              ) -> pd.DataFrame:
    """Match the merged pull requests to the nearest releases that include them."""
    assert isinstance(time_to, datetime)
    pr_releases = _new_map_df()
    if prs.empty:
        return pr_releases
    time_from = max(time_from, prs[PullRequest.merged_at.key].min() - timedelta(minutes=1))
    async with db.connection() as conn:
        repos = prs[PullRequest.repository_full_name.key].unique()
        releases = await load_releases(repos, time_from, time_to, release_settings, conn, cache)
        if cache is not None:
            matched_bys = {
                r: g.head(1).loc[0, matched_by_column] for r, g in releases[
                    [Release.repository_full_name.key, matched_by_column]
                ].groupby(Release.repository_full_name.key, sort=False, as_index=False)}
            pr_releases.append(await _load_pr_releases_from_cache(
                prs.index, prs[PullRequest.repository_full_name.key].values, matched_bys,
                release_settings, cache))
        merged_prs = prs[~prs.index.isin(pr_releases.index)]
        missed_releases = await _map_prs_to_releases(merged_prs, releases, conn, cache)
    if cache is not None:
        await _cache_pr_releases(missed_releases, release_settings, cache)
    return pr_releases.append(missed_releases)


index_name = "pull_request_node_id"


def _new_map_df(data=None) -> pd.DataFrame:
    columns = [Release.published_at.key,
               Release.author.key,
               Release.url.key,
               Release.repository_full_name.key,
               matched_by_column]
    if data is None:
        return pd.DataFrame(columns=columns, index=pd.Index([], name=index_name))
    return pd.DataFrame.from_records(data, columns=[index_name] + columns, index=index_name)


async def _load_pr_releases_from_cache(prs: Iterable[str],
                                       pr_repos: Iterable[str],
                                       matched_bys: Dict[str, int],
                                       release_settings: Dict[str, ReleaseMatchSetting],
                                       cache: aiomcache.Client) -> pd.DataFrame:
    batch_size = 32
    records = []
    utc = timezone.utc
    keys = [gen_cache_key("release_github|3|%s|%s", pr, release_settings["github.com/" + repo])
            for pr, repo in zip(prs, pr_repos)]
    for key, val in zip(keys, chain.from_iterable(
            [await cache.multi_get(*(k for _, k in g))
             for _, g in groupby(enumerate(keys), lambda ik: ik[0] // batch_size)])):
        if val is None:
            continue
        released_at, released_by, released_url, repo, matched_by = marshal.loads(val)
        if matched_by != matched_bys.get(repo, matched_by):
            continue
        released_at = datetime.fromtimestamp(released_at).replace(tzinfo=utc)
        records.append((key, released_at, released_by, released_url, repo, matched_by))
    df = _new_map_df(records)
    return df


async def _map_prs_to_releases(prs: pd.DataFrame,
                               releases: pd.DataFrame,
                               conn: databases.core.Connection,
                               cache: Optional[aiomcache.Client],
                               ) -> pd.DataFrame:
    releases = dict(list(releases.groupby(Release.repository_full_name.key, sort=False)))
    histories = await _fetch_release_histories(releases, conn, cache)
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
            r = repo_releases.xs(items[0])
            released_prs.append((pr_id,
                                 r[Release.published_at.key],
                                 r[Release.author.key],
                                 r[Release.url.key],
                                 repo,
                                 r[matched_by_column]))
    released_prs = _new_map_df(released_prs)
    released_prs[Release.published_at.key] = np.maximum(
        released_prs[Release.published_at.key],
        prs.loc[released_prs.index, PullRequest.merged_at.key])
    return postprocess_datetime(released_prs)


async def _fetch_release_histories(releases: Dict[str, pd.DataFrame],
                                   conn: databases.core.Connection,
                                   cache: Optional[aiomcache.Client],
                                   ) -> Dict[str, Dict[str, List[str]]]:
    histories = {}

    async def fetch_release_history(repo, repo_releases):
        histories[repo] = history = {}
        release_hashes = set(repo_releases[Release.sha.key].values)
        for rel_index, rel_commit_id, rel_sha in zip(
                repo_releases.index.values,
                repo_releases[Release.commit_id.key].values,
                repo_releases[Release.sha.key].values):
            if rel_sha in history:
                update_history(history, rel_sha, rel_index, release_hashes)
                continue
            dag = await _fetch_commit_history_dag(rel_commit_id, conn, cache)
            for c, edges in dag.items():
                items = history.get(c)
                if items is None:
                    history[c] = [rel_index, *edges]
                    continue
                items[0] = rel_index
                for v in edges:
                    if v not in items:
                        items.append(v)

    errors = await asyncio.gather(*(fetch_release_history(*r) for r in releases.items()),
                                  return_exceptions=True)
    for e in errors:
        if e is not None:
            raise e from None
    return histories


async def _cache_pr_releases(releases: pd.DataFrame,
                             release_settings: Dict[str, ReleaseMatchSetting],
                             cache: aiomcache.Client) -> None:
    mt = max_exptime
    for id, released_at, released_by, release_url, repo, matched_by in zip(
            releases.index, releases[Release.published_at.key],
            releases[Release.author.key].values, releases[Release.url.key].values,
            releases[Release.repository_full_name.key].values,
            releases[matched_by_column].values):
        key = gen_cache_key("release_github|3|%s|%s", id, release_settings["github.com/" + repo])
        t = released_at.timestamp(), released_by, release_url, repo, int(matched_by)
        await cache.set(key, marshal.dumps(t), exptime=mt)


@cached(
    exptime=24 * 60 * 60,  # 1 day
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda commit_id, **_: (commit_id,),
    refresh_on_access=True,
)
async def _fetch_first_parents(commit_id: str,
                               conn: databases.core.Connection,
                               cache: Optional[aiomcache.Client],
                               ) -> List[str]:
    # Git parent-child is reversed github_node_commit_parents' parent-child.
    quote = "`" if isinstance(conn.raw_connection, aiosqlite.core.Connection) else ""
    query = f"""
        WITH RECURSIVE commit_first_parents AS (
            SELECT
                p.child_id AS parent,
                cc.id AS parent_id
            FROM
                github_node_commit_parents p
                    LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                    LEFT JOIN github_node_commit cc ON p.child_id = cc.id
            WHERE
                p.parent_id = '{commit_id}' AND p.{quote}index{quote} = 0
            UNION
                SELECT
                    p.child_id AS parent,
                    cc.id AS parent_id
                FROM
                    github_node_commit_parents p
                        INNER JOIN commit_first_parents h ON h.parent = p.parent_id
                        LEFT JOIN github_node_commit pc ON p.parent_id = pc.id
                        LEFT JOIN github_node_commit cc ON p.child_id = cc.id
                WHERE p.{quote}index{quote} = 0
        ) SELECT
            parent_id
        FROM
            commit_first_parents;"""
    first_parents = [commit_id] + [r["parent_id"] for r in await conn.fetch_all(query)]
    return first_parents


@cached(
    exptime=max_exptime,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda commit_id, **_: (commit_id,),
    refresh_on_access=True,
)
async def _fetch_commit_history_dag(commit_id: str,
                                    conn: databases.core.Connection,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Dict[str, List[str]]:
    # How about limiting the history by some certain date?
    # That would speed up the things but then we'll not be able to cache it.
    # Credits: @dennwc
    # Git parent-child is reversed github_node_commit_parents' parent-child.
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
            p.parent_id = '{commit_id}'
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
    dag = {}
    if isinstance(conn.raw_connection, asyncpg.connection.Connection):
        # this works much faster then iterate() / fetch_all()
        rows = await conn.raw_connection.fetch(query)
    else:
        rows = await conn.fetch_all(query)
    for r in rows:
        # reverse the order so that parent-child matches github_node_commit_parents again
        child, parent = r
        try:
            dag[parent].append(child)
        except KeyError:
            # first iteration
            dag[parent] = [child]
        dag.setdefault(child, [])
    if not dag:
        # initial commit
        sha = await conn.fetch_val(select([PushCommit.sha]).where(PushCommit.node_id == commit_id))
        return {sha: []}
    return dag


async def _find_old_released_prs(releases: pd.DataFrame,
                                 time_boundary: datetime,
                                 db: DatabaseLike,
                                 cache: Optional[aiomcache.Client],
                                 ) -> pd.DataFrame:
    observed_commits, _, _ = await _extract_released_commits(releases, time_boundary, db, cache)
    repo = releases.iloc[0][Release.repository_full_name.key] if not releases.empty else ""
    return await read_sql_query(
        select([PullRequest])
        .where(and_(PullRequest.merged_at < time_boundary,
                    PullRequest.repository_full_name == repo,
                    PullRequest.merge_commit_sha.in_(observed_commits))),
        db, PullRequest, index=PullRequest.node_id.key)


async def _extract_released_commits(releases: pd.DataFrame,
                                    time_boundary: datetime,
                                    db: DatabaseLike,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Tuple[Dict[str, List[str]], pd.DataFrame, Dict[str, str]]:
    resolved_releases = set()
    full_dag = {}
    hash_to_release = {h: rid for rid, h in zip(releases.index, releases[Release.sha.key].values)}
    new_releases = releases[releases[Release.published_at.key] >= time_boundary]
    boundary_releases = set()
    for rid, commit_id, root in zip(new_releases.index,
                                    new_releases[Release.commit_id.key].values,
                                    new_releases[Release.sha.key].values):
        if rid in resolved_releases:
            continue
        dag = await _fetch_commit_history_dag(commit_id, db, cache)
        parents = [root]
        while parents:
            x = parents.pop()
            if x in full_dag:
                continue
            try:
                xrid = hash_to_release[x]
            except KeyError:
                pass
            else:
                pubdt = releases.loc[xrid, Release.published_at.key]
                if pubdt >= time_boundary:
                    resolved_releases.add(xrid)
                else:
                    boundary_releases.add(xrid)
                    continue
            children = dag[x]
            x_children = full_dag.setdefault(x, [])
            for c in children:
                if c not in x_children:
                    x_children.append(c)
            parents.extend(children)
    # we need to traverse full history from boundary_releases and subtract it from the full DAG
    ignored_commits = set()
    for rid in boundary_releases:
        release = releases.loc[rid]
        if release[Release.sha.key] in ignored_commits:
            continue
        dag = await _fetch_commit_history_dag(release[Release.commit_id.key], db, cache)
        parents = [release[Release.sha.key]]
        while parents:
            x = parents.pop()
            if x in ignored_commits:
                continue
            ignored_commits.add(x)
            children = dag[x]
            parents.extend(children)
    for c in ignored_commits:
        try:
            del full_dag[c]
        except KeyError:
            continue
    return full_dag, new_releases, hash_to_release


async def map_releases_to_prs(repos: Iterable[str],
                              time_from: datetime,
                              time_to: datetime,
                              release_settings: Dict[str, ReleaseMatchSetting],
                              db: DatabaseLike,
                              cache: Optional[aiomcache.Client],
                              ) -> pd.DataFrame:
    """Find pull requests which were released between `time_from` and `time_to` but merged before \
    `time_from`.

    :return: dataframe with found PRs.
    """
    assert isinstance(time_from, datetime)
    assert isinstance(time_to, datetime)
    old_from = time_from - timedelta(days=365)  # find PRs not older than 365 days before time_from
    releases = await load_releases(
        repos, old_from, time_to, release_settings, db, cache, index=Release.id.key)
    prs = []
    for _, repo_releases in releases.groupby(Release.repository_full_name.key, sort=False):
        prs.append(await _find_old_released_prs(repo_releases, time_from, db, cache))
    if prs:
        return pd.concat(prs, sort=False)
    return pd.DataFrame()


async def mine_releases(releases: pd.DataFrame,
                        time_boundary: datetime,
                        db: DatabaseLike,
                        cache: Optional[aiomcache.Client]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collect details about each release published after `time_boundary` and calculate added \
    and deleted line statistics."""
    miners = (_mine_monorepo_releases(repo_releases, time_boundary, db, cache)
              for _, repo_releases in releases.groupby(Release.repository_full_name.key,
                                                       sort=False))
    stats = await asyncio.gather(*miners, return_exceptions=True)
    for s in stats:
        if isinstance(s, BaseException):
            raise s from None
    user_columns = [User.login, User.avatar_url]
    if stats:
        stats = pd.concat(stats, sort=False)
        people = set(chain(chain.from_iterable(stats["commit_authors"]), stats["publisher"]))
        stats["publisher"] = "github.com/" + stats["publisher"]
        stats["repository"] = "github.com/" + stats["repository"]
        for calist in stats["commit_authors"].values:
            for i, v in enumerate(calist):
                calist[i] = "github.com/" + v
        avatars = await read_sql_query(
            select(user_columns).where(User.login.in_(people)), db, user_columns)
        avatars[User.login.key] = "github.com/" + avatars[User.login.key]
        return stats, avatars
    return pd.DataFrame(), pd.DataFrame(columns=[c.key for c in user_columns])


@cached(
    exptime=10 * 60,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda releases, **_: (sorted(releases.index),),
    refresh_on_access=True,
)
async def _mine_monorepo_releases(
        releases: pd.DataFrame,
        time_boundary: datetime,
        db: DatabaseLike,
        cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    dag, new_releases, hash_to_release = await _extract_released_commits(
        releases, time_boundary, db, cache)
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
            db, commit_df_columns)
        try:
            previous_published_at = max(releases.loc[hash_to_release[n], Release.published_at.key]
                                        for n in neighbours[sha] if n in hash_to_release)
        except ValueError:
            # no previous releases
            previous_published_at = await db.fetch_val(
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
