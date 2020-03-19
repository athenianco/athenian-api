from datetime import date, datetime, timedelta, timezone
from itertools import chain, groupby
import marshal
import pickle
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

import aiomcache
import databases
import pandas as pd
from sqlalchemy import and_, desc, func, select

from athenian.api.async_read_sql_query import postprocess_datetime, read_sql_query
from athenian.api.cache import cached, max_exptime
from athenian.api.models.metadata.github import NodeCommit, PullRequest, PullRequestMergeCommit, \
    Release


async def map_prs_to_releases(prs: pd.DataFrame,
                              time_to: date,
                              db: databases.Database,
                              cache: Optional[aiomcache.Client],
                              ) -> pd.DataFrame:
    """Match the merged pull requests to the nearest releases that include them."""
    releases = _new_map_df()
    if prs.empty:
        return releases
    if cache is not None:
        releases.append(
            await _load_pr_releases_from_cache(prs.index, cache))
    merged_prs = prs[~prs.index.isin(releases.index)]
    missed_releases = await _map_prs_to_releases(merged_prs, time_to, db, cache)
    if cache is not None:
        await _cache_pr_releases(missed_releases, cache)
    return releases.append(missed_releases)


index_name = "pull_request_node_id"


def _new_map_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[Release.published_at.key, Release.author.key, Release.url.key],
                        index=pd.Index([], name=index_name))


async def _load_pr_releases_from_cache(
        prs: Iterable[str], cache: aiomcache.Client) -> pd.DataFrame:
    batch_size = 32
    df = _new_map_df()
    utc = timezone.utc
    keys = [b"release_github|" + pr.encode() for pr in prs]
    for key, val in zip(keys, chain.from_iterable(
            [await cache.multi_get(*(k for _, k in g))
             for _, g in groupby(enumerate(keys), lambda ik: ik[0] // batch_size)])):
        if val is None:
            continue
        released_at, released_by, released_url = marshal.loads(val)
        released_at = datetime.fromtimestamp(released_at).replace(tzinfo=utc)
        df.loc[key] = released_at, released_by, released_url
    return df


async def _map_prs_to_releases(prs: pd.DataFrame,
                               time_to: date,
                               db: databases.Database,
                               cache: Optional[aiomcache.Client],
                               ) -> pd.DataFrame:
    time_from = prs[PullRequest.merged_at.key].min()
    time_to = pd.Timestamp(time_to, tzinfo=timezone.utc)
    prrfnkey = PullRequest.repository_full_name.key
    async with db.connection() as conn:
        merge_hashes = await read_sql_query(
            select([PullRequestMergeCommit]).where(PullRequestMergeCommit.id.in_(prs.index)),
            conn, PullRequestMergeCommit, index=PullRequestMergeCommit.id.key)
        prs_by_repo = prs.merge(merge_hashes, left_index=True, right_index=True)
        prs_by_repo.index.name = prs.index.name
        prs_by_repo.reset_index(inplace=True)
        prs_by_repo.set_index([prrfnkey, PullRequest.node_id.key], inplace=True)
        releases = await read_sql_query(
            select([Release])
            .where(and_(Release.published_at.between(time_from, time_to),
                        Release.repository_full_name.in_(prs[prrfnkey].unique())))
            .order_by(Release.published_at),
            conn, Release)
        return await _map_prs_to_specific_releases(prs_by_repo, releases, conn, cache)


async def _map_prs_to_specific_releases(merged_prs_by_repo: pd.DataFrame,
                                        releases: pd.DataFrame,
                                        conn: databases.core.Connection,
                                        cache: Optional[aiomcache.Client],
                                        ) -> pd.DataFrame:
    rrfnkey = Release.repository_full_name.key
    released_prs = _new_map_df()
    pr_id_by_hash_by_repo = merged_prs_by_repo[[PullRequestMergeCommit.sha.key]].copy()
    pr_id_by_hash_by_repo.reset_index(inplace=True)
    pr_id_by_hash_by_repo.set_index(
        [PullRequest.repository_full_name.key, PullRequestMergeCommit.sha.key], inplace=True)
    pr_id_by_hash_by_repo = pr_id_by_hash_by_repo[PullRequest.node_id.key]  # pd.Series
    # TODO(vmarkovtsev): we don't actually have to fetch the history for all the releases because
    # later releases are going to include earlier releases.
    for repo, repo_releases in releases.groupby(rrfnkey, sort=False):
        repo_releases = repo_releases.iloc[:-1]
        pr_id_by_hash = pr_id_by_hash_by_repo[repo]
        backlog = set(pr_id_by_hash.index)
        for relc, relsha, reldate, relauthor, relurl in zip(
                repo_releases[Release.commit_id.key].values,
                repo_releases[Release.sha.key].values,
                repo_releases[Release.published_at.key].values,
                repo_releases[Release.author.key].values,
                repo_releases[Release.url.key].values,
        ):
            history = await _fetch_commit_history_set(relc, relsha, conn, cache)
            for merge_hash in history.intersection(backlog):
                released_prs.loc[pr_id_by_hash[merge_hash]] = reldate, relauthor, relurl
                backlog.remove(merge_hash)
            if not backlog:
                break
    return postprocess_datetime(released_prs)


async def _cache_pr_releases(releases: pd.DataFrame, cache: aiomcache.Client) -> None:
    mt = max_exptime
    for id, released_at, released_by, release_url in zip(
            releases.index, releases[Release.published_at.key],
            releases[Release.author.key].values, releases[Release.url.key].values):
        await cache.set(b"release_github|" + id.encode(),
                        marshal.dumps((released_at.timestamp(), released_by, release_url)),
                        exptime=mt)


@cached(
    exptime=max_exptime,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda commit_id, **_: (commit_id,),
)
async def _fetch_commit_history_set(commit_id: str,
                                    commit_sha: str,
                                    conn: databases.core.Connection,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Set[str]:
    # How about limiting the history by some certain date?
    # That would speed up the things but then we'll not be able to cache it.
    # Credits: @dennwc
    # Git parent-child is reversed github_node_commit_parents' parent-child.
    query = f"""
    WITH RECURSIVE commit_history AS (
        SELECT
            p.child_id AS parent,
            c.oid AS parent_oid
        FROM
            github_node_commit_parents p
                LEFT JOIN github_node_commit c ON p.child_id = c.id
        WHERE
            p.parent_id = '{commit_id}'
        UNION
            SELECT
                p.child_id AS parent,
                c.oid AS parent_oid
            FROM
                github_node_commit_parents p
                    INNER JOIN commit_history h ON h.parent = p.parent_id
                    LEFT JOIN github_node_commit c ON p.child_id = c.id
    ) SELECT
        parent_oid
    FROM
        commit_history;"""
    history = {r["parent_oid"] for r in await conn.fetch_all(query)}
    history.add(commit_sha)
    return history


@cached(
    exptime=max_exptime,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda commit_id, **_: (commit_id,),
)
async def _fetch_commit_history_dag(commit_id: str,
                                    conn: databases.core.Connection,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Tuple[str, Dict[str, List[str]]]:
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
    history = [(r["child_oid"], r["parent_oid"]) for r in await conn.fetch_all(query)]
    # parent-child matches github_node_commit_parents again
    root = history[0][0]
    dag = {}  # not defaultdict to enable marshal
    for p, c in history:
        try:
            dag[p].append(c)
        except KeyError:
            dag[p] = [c]
    return root, dag


async def _find_old_released_prs(releases: pd.DataFrame,
                                 time_boundary: pd.Timestamp,
                                 db: Union[databases.Database, databases.core.Connection],
                                 cache: Optional[aiomcache.Client],
                                 ) -> pd.DataFrame:
    resolved = set()
    observed_commits = set()
    hash_to_release = {h: rid for rid, h in zip(releases.index, releases[Release.sha.key].values)}
    new_releases = releases[releases[Release.published_at.key] >= time_boundary]
    boundary_releases = set()
    for rid, commit_id in zip(new_releases.index, new_releases[Release.commit_id.key].values):
        if rid in resolved:
            continue
        root, dag = await _fetch_commit_history_dag(commit_id, db, cache)
        parents = [root]
        while parents:
            x = parents.pop()
            if x in observed_commits:
                continue
            try:
                xrid = hash_to_release[x]
            except KeyError:
                pass
            else:
                pubdt = releases.loc[xrid][Release.published_at.key]
                if pubdt >= time_boundary:
                    resolved.add(xrid)
                else:
                    boundary_releases.add(xrid)
                    continue
            observed_commits.add(x)
            children = dag.get(x, [])
            parents.extend(children)
    # we need to traverse full history from boundary_releases and subtract it from observed_commits
    released_commits = set()
    for rid in boundary_releases:
        if releases.loc[rid][Release.sha.key] in released_commits:
            continue
        root, dag = await _fetch_commit_history_dag(
            releases.loc[rid][Release.commit_id.key], db, cache)
        parents = [root]
        while parents:
            x = parents.pop()
            if x in released_commits:
                continue
            released_commits.add(x)
            children = dag.get(x, [])
            parents.extend(children)
    observed_commits -= released_commits
    repo = releases.iloc[0][Release.repository_full_name.key] if not releases.empty else ""
    return await read_sql_query(
        select([PullRequest])
        .where(and_(PullRequest.merged_at < time_boundary,
                    PullRequest.repository_full_name == repo,
                    PullRequest.merge_commit_sha.in_(observed_commits))),
        db, PullRequest, index=PullRequest.node_id.key)


async def map_releases_to_prs(repos: Iterable[str],
                              time_from: date,
                              time_to: date,
                              db: Union[databases.Database, databases.core.Connection],
                              cache: Optional[aiomcache.Client],
                              ) -> pd.DataFrame:
    """Find pull requests which were released between `time_from` and `time_to` but merged before \
    `time_from`.

    :return: dataframe with found PRs.
    """
    time_from = pd.Timestamp(time_from, tzinfo=timezone.utc)
    time_to = pd.Timestamp(time_to, tzinfo=timezone.utc)
    old_from = time_from - timedelta(days=365)  # find PRs not older than 365 days before time_from
    releases = await read_sql_query(select([Release])
                                    .where(and_(Release.repository_full_name.in_(repos),
                                                Release.published_at.between(old_from, time_to)))
                                    .order_by(desc(Release.published_at)),
                                    db, Release, index=Release.id.key)
    prs = []
    for _, repo_releases in releases.groupby(Release.repository_full_name.key, sort=False):
        prs.append(await _find_old_released_prs(repo_releases, time_from, db, cache))
    return pd.concat(prs, sort=False)


@cached(
    exptime=max_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repo, min_published_at, **_: (repo, min_published_at),
)
async def _fetch_release_by_timestamp(repo: str,
                                      min_published_at: datetime,
                                      db: Union[databases.Database, databases.core.Connection],
                                      cache: Optional[aiomcache.Client]) -> Dict[str, Any]:
    return dict(await db.fetch_one(
        select([Release.sha, Release.commit_id, Release.author, Release.url])
        .where(and_(Release.repository_full_name == repo,
                    Release.published_at == min_published_at))))


@cached(
    exptime=max_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda release, **_: (release[Release.sha.key], release[Release.repository_full_name.key]),
)
async def _fetch_diff_commit_history(release: Mapping[str, Any],
                                     db: Union[databases.Database, databases.core.Connection],
                                     cache: Optional[aiomcache.Client]):
    repo = release[Release.repository_full_name.key]
    rel_history = await _fetch_commit_history_set(
        release[Release.commit_id.key], release[Release.sha.key], db, cache)
    prev_published_at = await db.fetch_val(
        select([func.max(Release.published_at)])
        .where(and_(Release.repository_full_name == repo,
                    Release.published_at < release[Release.published_at.key])))
    if prev_published_at is not None:
        prev_commit = await db.fetch_one(
            select([Release.commit_id, Release.sha])
            .where(and_(Release.repository_full_name == repo,
                        Release.published_at == prev_published_at)))
        prev_history = await _fetch_commit_history_set(*prev_commit, db, cache)
        diff_history = rel_history - prev_history
        min_commit_date = await db.fetch_val(
            select([func.min(NodeCommit.pushed_date)]).where(NodeCommit.oid.in_(diff_history)))
    else:
        diff_history = rel_history
        min_commit_date = datetime(year=1970, month=1, day=1)
    return diff_history, min_commit_date
