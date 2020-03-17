from collections import defaultdict
from datetime import date, datetime, timezone
from itertools import chain, groupby, repeat
import marshal
import pickle
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

import aiomcache
import databases
import pandas as pd
from sqlalchemy import and_, desc, func, join, select

from athenian.api.async_read_sql_query import read_sql_query
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
    if cache is not None:
        releases.append(
            await _load_pr_releases_from_cache(prs.index, cache))
    merged_prs = prs[~prs.index.isin(releases.index)]
    missed_releases = await _map_prs_to_releases(merged_prs, time_to, db, cache)
    if cache is not None:
        await _cache_pr_releases(missed_releases, cache)
    return releases.append(missed_releases)


column_released_at = "released_at"
column_released_by = "released_by"


def _new_map_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[column_released_at, column_released_by],
                        index=pd.Index([], name="pull_request_node_id"))


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
        released_at, released_by = marshal.loads(val)
        released_at = datetime.fromtimestamp(released_at).replace(tzinfo=utc)
        df.loc[key] = released_at, released_by
    return df


async def _map_prs_to_releases(prs: pd.DataFrame,
                               time_to: date,
                               db: databases.Database,
                               cache: Optional[aiomcache.Client],
                               ) -> pd.DataFrame:
    time_from = prs[PullRequest.merged_at.key].min()
    prrfnkey = PullRequest.repository_full_name.key
    repos = prs[prrfnkey].unique()
    rrfnkey = Release.repository_full_name.key
    async with db.connection() as conn:
        merge_hashes = await read_sql_query(
            select([PullRequestMergeCommit]).where(PullRequestMergeCommit.id.in_(prs.index)),
            conn, PullRequestMergeCommit, index=PullRequestMergeCommit.id.key)
        prs_by_repo = prs.merge(merge_hashes, left_index=True, right_index=True)
        prs_by_repo.reset_index(inplace=True)
        prs_by_repo.set_index([prrfnkey, PullRequest.node_id.key], inplace=True)
        releases = await read_sql_query(
            select([Release])
            .where(and_(Release.published_at.between(time_from, time_to),
                        Release.repository_full_name.in_(repos)))
            .order_by(desc(Release.published_at)),
            conn, Release)
        latest_releases = releases.drop_duplicates([rrfnkey])
        released_prs: Dict[str, List[Tuple[datetime, str]]] = {}  # PRs released before `time_to`
        # Dict value: list of length 2 with the lower and the upper release time bounds and
        # corresponding release authors.
        repos_left = defaultdict(set)
        pr_by_sha: Dict[str, str] = {}
        for repo, relc, relsha, reldate, relauthor in zip(
                latest_releases[rrfnkey].values,
                latest_releases[Release.commit_id.key].values,
                latest_releases[Release.sha.key].values,
                latest_releases[Release.published_at.key].values,
                latest_releases[Release.author.key].values,
        ):
            history = await _fetch_commit_history(relc, relsha, conn, cache)
            repo_prs = prs_by_repo.loc[repo]
            for pr_id, pr_sha in zip(repo_prs.index,
                                     repo_prs[PullRequestMergeCommit.sha.key].values):
                if pr_sha in history:
                    released_prs[pr_id] = [(None, None), (reldate, relauthor)]
                    pr_by_sha[pr_sha] = pr_id
                    repos_left[repo].add(pr_sha)
        # In theory, we could run some clever binary search to skip fetching some release
        # histories. In practice, however, every release is going to map to some PRs.
        # Thus we fetch each release until there are unmatched PRs.
        for repo, repo_releases in releases.groupby(rrfnkey, sort=False):
            repo_releases = repo_releases.iloc[1::-1]
            backlog = repos_left[repo]
            # We have already fetched the first, the latest release.
            # We will fetch release histories from the earliest to the latest.
            for relc, relsha, reldate, relauthor in zip(
                    repo_releases[Release.commit_id.key].values,
                    repo_releases[Release.sha.key].values,
                    repo_releases[Release.published_at.key].values,
                    latest_releases[Release.author.key].values,
            ):
                history = await _fetch_commit_history(relc, relsha, conn, cache)
                for merge_hash in history.intersection(backlog):
                    released_prs[pr_by_sha[merge_hash]][0] = (reldate, relauthor)
                    backlog.remove(merge_hash)
                if not backlog:
                    break
            for merge_hash in backlog:
                # Set the lower bound of the remaining PRs to the upper bound.
                bounds = released_prs[pr_by_sha[merge_hash]]
                bounds[0] = bounds[1]
    result = _new_map_df()
    for pr, ((reldate, relauthor), _) in released_prs.items():
        result.loc[pr] = reldate, relauthor
    return result


async def _cache_pr_releases(releases: pd.DataFrame, cache: aiomcache.Client) -> None:
    mt = max_exptime
    for id, released_at, released_by in zip(releases.index,
                                            releases[column_released_at],
                                            releases[column_released_by].values):
        await cache.set(b"release_github|" + id.encode(),
                        marshal.dumps((released_at.timestamp(), released_by)),
                        exptime=mt)


@cached(
    exptime=max_exptime,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda commit_id, **_: (commit_id,),
)
async def _fetch_commit_history(commit_id: str,
                                commit_sha: str,
                                conn: databases.core.Connection,
                                cache: Optional[aiomcache.Client],
                                ) -> Set[str]:
    # Credits: @dennwc
    query = f"""
WITH RECURSIVE commit_history AS (
    SELECT
        parent_id AS child,
        child_id AS parent,
        cc.oid AS child_oid,
        pc.oid AS parent_oid
    FROM
        github_node_commit_parents p
            LEFT JOIN github_node_commit cc ON p.parent_id = cc.id
            LEFT JOIN github_node_commit pc ON p.child_id = pc.id
    WHERE
        parent_id = '{commit_id}'
    UNION
        SELECT
            parent_id AS child,
            child_id AS parent,
            cc.oid AS child_oid,
            pc.oid AS parent_oid
        FROM
            github_node_commit_parents e
                INNER JOIN commit_history s ON s.parent = e.parent_id
                LEFT JOIN github_node_commit cc ON e.parent_id = cc.id
                LEFT JOIN github_node_commit pc ON e.child_id = pc.id
) SELECT
    parent_oid AS sha
FROM
    commit_history;"""
    history = {r["sha"] for r in await conn.fetch_all(query)}
    history.add(commit_sha)
    return history


async def map_releases_to_prs(repos: Iterable[str],
                              time_from: date,
                              time_to: date,
                              db: Union[databases.Database, databases.core.Connection],
                              cache: Optional[aiomcache.Client],
                              ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Find pull requests which were released between `time_from` and `time_to` but merged before \
    `time_from`.

    :return: dataframe with PRs, dataframe with the corresponding release mapping.
    """
    minrows = await db.fetch_all(
        select([Release.repository_full_name, func.min(Release.published_at)])
        .where(and_(Release.repository_full_name.in_(repos),
                    Release.published_at.between(time_from, time_to)))
        .group_by(Release.repository_full_name))
    prs = []
    releases = []
    for repo, min_published_at in minrows:
        release = dict(await _fetch_release_by_timestamp(repo, min_published_at, db, cache))
        release[Release.published_at.key] = min_published_at
        release[Release.repository_full_name.key] = repo
        diff_history, min_commit_date = await _fetch_diff_commit_history(release, db, cache)
        repo_prs = await read_sql_query(
            select([PullRequest])
            .select_from(join(PullRequest, PullRequestMergeCommit,
                              PullRequest.node_id == PullRequestMergeCommit.id))
            .where(and_(PullRequest.merged_at.between(min_commit_date,
                                                      min(min_published_at, time_from)),
                        PullRequest.repository_full_name == repo,
                        PullRequestMergeCommit.sha.in_(diff_history))),
            db, PullRequest, index=PullRequest.node_id.key)
        prs.append(repo_prs)
        repo_rels = pd.DataFrame.from_records(
            repeat((min_published_at, release[Release.author.key]), len(repo_prs)),
            index=repo_prs.index, columns=[column_released_at, column_released_by])
        repo_rels.index.name = "pull_request_node_id"
        releases.append(repo_rels)
    if not prs:
        return None, None
    return pd.concat(prs), pd.concat(releases)


@cached(
    exptime=max_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repo, min_published_at, **_: (repo, min_published_at),
)
async def _fetch_release_by_timestamp(repo: str,
                                      min_published_at: datetime,
                                      db: Union[databases.Database, databases.core.Connection],
                                      cache: Optional[aiomcache.Client]):
    return await db.fetch_one(select([Release.sha, Release.commit_id, Release.author])
                              .where(and_(Release.repository_full_name == repo,
                                          Release.published_at == min_published_at)))


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
    rel_history = await _fetch_commit_history(
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
        prev_history = await _fetch_commit_history(*prev_commit, db, cache)
        diff_history = rel_history - prev_history
        min_commit_date = await db.fetch_val(
            select([func.min(NodeCommit.pushed_date)]).where(NodeCommit.oid.in_(diff_history)))
    else:
        diff_history = rel_history
        min_commit_date = datetime(year=1970, month=1, day=1)
    return diff_history, min_commit_date
