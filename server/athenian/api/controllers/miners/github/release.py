from datetime import date, datetime, timedelta, timezone
from itertools import chain, groupby
import marshal
from typing import Dict, Iterable, List, Optional, Union

import aiomcache
import databases
import pandas as pd
from sqlalchemy import and_, desc, select

from athenian.api.async_read_sql_query import postprocess_datetime, read_sql_query
from athenian.api.cache import cached, max_exptime
from athenian.api.models.metadata.github import PullRequest, Release


async def map_prs_to_releases(prs: pd.DataFrame,
                              time_to: datetime,
                              db: databases.Database,
                              cache: Optional[aiomcache.Client],
                              ) -> pd.DataFrame:
    """Match the merged pull requests to the nearest releases that include them."""
    assert isinstance(time_to, datetime)
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
    async with db.connection() as conn:
        repos = prs[PullRequest.repository_full_name.key].unique()
        releases = await read_sql_query(
            select([Release])
            .where(and_(Release.published_at.between(time_from, time_to),
                        Release.repository_full_name.in_(repos)))
            .order_by(desc(Release.published_at)),
            conn, Release)
        releases = dict(list(releases.groupby(Release.repository_full_name.key, sort=False)))
        histories = await _fetch_release_histories(releases, conn, cache)
        released_prs = _new_map_df()
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
                r = repo_releases.iloc[items[0]]
                released_prs.loc[pr_id] = (r[Release.published_at.key],
                                           r[Release.author.key],
                                           r[Release.url.key])
        return postprocess_datetime(released_prs)


async def _fetch_release_histories(releases: Dict[str, pd.DataFrame],
                                   conn: databases.core.Connection,
                                   cache: Optional[aiomcache.Client],
                                   ) -> Dict[str, Dict[str, List[str]]]:
    histories = {}
    for repo, repo_releases in releases.items():
        histories[repo] = history = {}
        release_hashes = set(repo_releases[Release.sha.key].values)
        for rel_index, (rel_commit_id, rel_sha) in enumerate(zip(
                repo_releases[Release.commit_id.key].values,
                repo_releases[Release.sha.key].values)):
            if rel_sha in history:
                parents = [rel_sha]
                while parents:
                    x = parents.pop()
                    if history[x][0] == rel_index or (x in release_hashes and x != rel_sha):
                        continue
                    history[x][0] = rel_index
                    parents.extend(history[x][1:])
                continue
            dag = await _fetch_commit_history_dag(rel_commit_id, conn, cache)
            for c, edges in dag.items():
                try:
                    items = history[c]
                except KeyError:
                    history[c] = [rel_index] + edges
                    continue
                items[0] = rel_index
                for v in edges:
                    if v not in items:
                        items.append(v)
    return histories


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
    history = [(r["child_oid"], r["parent_oid"]) for r in await conn.fetch_all(query)]
    # parent-child matches github_node_commit_parents again
    dag = {history[0][0]: []}
    for p, c in history:
        dag[p].append(c)
        dag.setdefault(c, [])
    return dag


async def _find_old_released_prs(releases: pd.DataFrame,
                                 time_boundary: datetime,
                                 db: Union[databases.Database, databases.core.Connection],
                                 cache: Optional[aiomcache.Client],
                                 ) -> pd.DataFrame:
    resolved = set()
    observed_commits = set()
    hash_to_release = {h: rid for rid, h in zip(releases.index, releases[Release.sha.key].values)}
    new_releases = releases[releases[Release.published_at.key] >= time_boundary]
    boundary_releases = set()
    for rid, commit_id, root in zip(new_releases.index,
                                    new_releases[Release.commit_id.key].values,
                                    new_releases[Release.sha.key].values):
        if rid in resolved:
            continue
        dag = await _fetch_commit_history_dag(commit_id, db, cache)
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
            children = dag[x]
            parents.extend(children)
    # we need to traverse full history from boundary_releases and subtract it from observed_commits
    released_commits = set()
    for rid in boundary_releases:
        release = releases.loc[rid]
        if release[Release.sha.key] in released_commits:
            continue
        dag = await _fetch_commit_history_dag(release[Release.commit_id.key], db, cache)
        parents = [release[Release.sha.key]]
        while parents:
            x = parents.pop()
            if x in released_commits:
                continue
            released_commits.add(x)
            children = dag[x]
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
                              time_from: datetime,
                              time_to: datetime,
                              db: Union[databases.Database, databases.core.Connection],
                              cache: Optional[aiomcache.Client],
                              ) -> pd.DataFrame:
    """Find pull requests which were released between `time_from` and `time_to` but merged before \
    `time_from`.

    :return: dataframe with found PRs.
    """
    assert isinstance(time_from, datetime)
    assert isinstance(time_to, datetime)
    old_from = time_from - timedelta(days=365)  # find PRs not older than 365 days before time_from
    releases = await read_sql_query(select([Release])
                                    .where(and_(Release.repository_full_name.in_(repos),
                                                Release.published_at.between(old_from, time_to)))
                                    .order_by(desc(Release.published_at)),
                                    db, Release, index=Release.id.key)
    prs = []
    for _, repo_releases in releases.groupby(Release.repository_full_name.key, sort=False):
        prs.append(await _find_old_released_prs(repo_releases, time_from, db, cache))
    if prs:
        return pd.concat(prs, sort=False)
    return pd.DataFrame()
