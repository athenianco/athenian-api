from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
import pickle
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Tuple

import aiomcache
import asyncpg
import databases
import lz4.frame
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, desc, insert, outerjoin, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.github.dag_accelerated import extract_first_parents, \
    extract_subdag, join_dags, partition_dag, searchsorted_inrange
from athenian.api.db import add_pdb_hits, add_pdb_misses
from athenian.api.defer import defer
from athenian.api.models.metadata.github import NodeCommit, NodePullRequestCommit, PushCommit, User
from athenian.api.models.precomputed.models import GitHubCommitHistory
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import DatabaseLike


class FilterCommitsProperty(Enum):
    """Primary commit filter modes."""

    NO_PR_MERGES = "no_pr_merges"
    BYPASSING_PRS = "bypassing_prs"


@sentry_span
@cached(
    exptime=5 * 60,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda prop, date_from, date_to, repos, with_author, with_committer, **kwargs:
        (prop.value, date_from.timestamp(), date_to.timestamp(), ",".join(sorted(repos)),
         ",".join(sorted(with_author)) if with_author else "",
         ",".join(sorted(with_committer)) if with_committer else "",
         "" if kwargs.get("columns") is None else ",".join(c.key for c in kwargs["columns"])),
)
async def extract_commits(prop: FilterCommitsProperty,
                          date_from: datetime,
                          date_to: datetime,
                          repos: Collection[str],
                          with_author: Optional[Collection[str]],
                          with_committer: Optional[Collection[str]],
                          meta_ids: Tuple[int, ...],
                          mdb: DatabaseLike,
                          cache: Optional[aiomcache.Client],
                          columns: Optional[List[InstrumentedAttribute]] = None,
                          ):
    """Fetch commits that satisfy the given filters."""
    assert isinstance(date_from, datetime)
    assert isinstance(date_to, datetime)
    log = logging.getLogger("%s.extract_commits" % metadata.__package__)
    sql_filters = [
        PushCommit.acc_id.in_(meta_ids),
        PushCommit.committed_date.between(date_from, date_to),
        PushCommit.repository_full_name.in_(repos),
        PushCommit.committer_email != "noreply@github.com",
    ]
    user_logins = set()
    if with_author:
        user_logins.update(with_author)
    if with_committer:
        user_logins.update(with_committer)
    if user_logins:
        rows = await mdb.fetch_all(
            select([User.login, User.node_id])
            .where(and_(User.login.in_(user_logins), User.acc_id.in_(meta_ids))))
        user_ids = {r[0]: r[1] for r in rows}
        del user_logins
    else:
        user_ids = {}
    if with_author:
        author_ids = []
        for u in with_author:
            try:
                author_ids.append(user_ids[u])
            except KeyError:
                continue
        sql_filters.append(PushCommit.author_user.in_(author_ids))
    if with_committer:
        committer_ids = []
        for u in with_committer:
            try:
                committer_ids.append(user_ids[u])
            except KeyError:
                continue
        sql_filters.append(PushCommit.committer_user.in_(committer_ids))
    if columns is None:
        cols_query, cols_df = [PushCommit], PushCommit
    else:
        if PushCommit.node_id not in columns:
            columns.append(PushCommit.node_id)
        cols_query = cols_df = columns
    if prop == FilterCommitsProperty.NO_PR_MERGES:
        with sentry_sdk.start_span(op="extract_commits/fetch/NO_PR_MERGES"):
            commits = await read_sql_query(
                select(cols_query).where(and_(*sql_filters)), mdb, cols_df)
    elif prop == FilterCommitsProperty.BYPASSING_PRS:
        with sentry_sdk.start_span(op="extract_commits/fetch/BYPASSING_PRS"):
            commits = await read_sql_query(
                select(cols_query)
                .select_from(outerjoin(PushCommit, NodePullRequestCommit,
                                       and_(PushCommit.node_id == NodePullRequestCommit.commit,
                                            PushCommit.acc_id == NodePullRequestCommit.acc_id)))
                .where(and_(NodePullRequestCommit.commit.is_(None), *sql_filters)),
                mdb, cols_df)
    else:
        raise AssertionError('Unsupported primary commit filter "%s"' % prop)
    for number_prop in (PushCommit.additions, PushCommit.deletions, PushCommit.changed_files):
        try:
            number_col = commits[number_prop.key]
        except KeyError:
            continue
        nans = commits[PushCommit.node_id.key].take(np.where(number_col.isna())[0])
        if not nans.empty:
            log.error("[DEV-546] Commits have NULL in %s: %s", number_prop.key, ", ".join(nans))
    return commits


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, branches, columns, prune, **_: (
        ",".join(sorted(repos)),
        ",".join(np.sort(branches[columns[0]].values)),
        prune,
    ),
    refresh_on_access=True,
)
async def fetch_repository_commits(repos: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                   branches: pd.DataFrame,
                                   columns: Tuple[str, str, str, str],
                                   prune: bool,
                                   meta_ids: Tuple[int, ...],
                                   mdb: databases.Database,
                                   pdb: databases.Database,
                                   cache: Optional[aiomcache.Client],
                                   ) -> Dict[str, Tuple[np.ndarray, np.array, np.array]]:
    """
    Load full commit DAGs for the given repositories.

    :param repos: Map from repository names to their precomputed DAGs.
    :param branches: Commits must contain all the existing commits in this DataFrame.
    :param columns: Names of the columns in `branches` that correspond to: \
                    1. Commit hash. \
                    2. Commit node ID. \
                    3. Commit timestamp. \
                    4. Commit repository name.
    :param prune: Remove any commits that are not accessible from `branches`.
    :return: Map from repository names to their DAGs.
    """
    missed_counter = 0
    repo_heads = {}
    sha_col, id_col, dt_col, repo_col = columns
    hash_to_id = dict(zip(branches[sha_col].values, branches[id_col].values))
    hash_to_dt = dict(zip(branches[sha_col].values, branches[dt_col].values))
    result = {}
    tasks = []
    for repo, repo_df in branches[[repo_col, sha_col]].groupby(repo_col, sort=False):
        required_heads = repo_df[sha_col].values.astype("U40")
        repo_heads[repo] = required_heads
        hashes, vertexes, edges = repos[repo]
        if len(hashes) > 0:
            found_indexes = searchsorted_inrange(hashes, required_heads)
            missed_mask = hashes[found_indexes] != required_heads
            missed_counter += missed_mask.sum()
            missed_heads = required_heads[missed_mask]  # these hashes do not exist in the p-DAG
        else:
            missed_heads = required_heads
        if len(missed_heads) > 0:
            # heuristic: order the heads from most to least recent
            order = sorted([(hash_to_dt[h], i) for i, h in enumerate(missed_heads)], reverse=True)
            missed_heads = [missed_heads[i] for _, i in order]
            missed_ids = [hash_to_id[h] for h in missed_heads]
            tasks.append(_fetch_commit_history_dag(
                hashes, vertexes, edges, missed_heads, missed_ids, repo, meta_ids, mdb))
        else:
            if prune:
                hashes, vertexes, edges = extract_subdag(hashes, vertexes, edges, required_heads)
            result[repo] = hashes, vertexes, edges
    # traverse commits starting from the missing branch heads
    add_pdb_hits(pdb, "fetch_repository_commits", len(branches) - missed_counter)
    add_pdb_misses(pdb, "fetch_repository_commits", missed_counter)
    if tasks:
        new_dags = await gather(*tasks, op="fetch_repository_commits/mdb")
        sql_values = []
        for repo, hashes, vertexes, edges in new_dags:
            assert (hashes[1:] > hashes[:-1]).all(), repo
            sql_values.append(GitHubCommitHistory(
                repository_full_name=repo,
                dag=lz4.frame.compress(pickle.dumps((hashes, vertexes, edges))),
            ).create_defaults().explode(with_primary_keys=True))
            if prune:
                hashes, vertexes, edges = extract_subdag(hashes, vertexes, edges, repo_heads[repo])
            result[repo] = hashes, vertexes, edges
        if pdb.url.dialect in ("postgres", "postgresql"):
            sql = postgres_insert(GitHubCommitHistory)
            sql = sql.on_conflict_do_update(
                constraint=GitHubCommitHistory.__table__.primary_key,
                set_={GitHubCommitHistory.dag.key: sql.excluded.dag,
                      GitHubCommitHistory.updated_at.key: sql.excluded.updated_at})
        elif pdb.url.dialect == "sqlite":
            sql = insert(GitHubCommitHistory).prefix_with("OR REPLACE")
        else:
            raise AssertionError("Unsupported database dialect: %s" % pdb.url.dialect)
        await defer(pdb.execute_many(sql, sql_values), "fetch_repository_commits/pdb")
    for repo, pdag in repos.items():
        if repo not in result:
            result[repo] = _empty_dag() if prune else pdag
    return result


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, **_: (",".join(sorted(repos)),),
)
async def fetch_precomputed_commit_history_dags(
        repos: Iterable[str],
        pdb: databases.Database,
        cache: Optional[aiomcache.Client],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load commit DAGs from the pdb."""
    ghrc = GitHubCommitHistory
    with sentry_sdk.start_span(op="fetch_precomputed_commit_history_dags/pdb"):
        rows = await pdb.fetch_all(
            select([ghrc.repository_full_name, ghrc.dag])
            .where(and_(
                ghrc.format_version == ghrc.__table__.columns[ghrc.format_version.key].default.arg,
                ghrc.repository_full_name.in_(repos),
            )))
    dags = {row[0]: pickle.loads(lz4.frame.decompress(row[1])) for row in rows}
    for repo in repos:
        if repo not in dags:
            dags[repo] = _empty_dag()
    return dags


def _empty_dag() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.array([], dtype="U40"), np.array([0], dtype=np.uint32), np.array([], dtype=np.uint32)


@sentry_span
async def _fetch_commit_history_dag(hashes: np.ndarray,
                                    vertexes: np.ndarray,
                                    edges: np.ndarray,
                                    head_hashes: Sequence[str],
                                    head_ids: Sequence[str],
                                    repo: str,
                                    meta_ids: Tuple[int, ...],
                                    mdb: databases.Database,
                                    ) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    max_stop_heads = 25
    max_inner_partitions = 25
    # Find max `max_stop_heads` top-level most recent commit hashes.
    stop_heads = hashes[np.delete(np.arange(len(hashes)), np.unique(edges))]
    if len(stop_heads) > 0:
        if len(stop_heads) > max_stop_heads:
            min_commit_time = datetime.now(timezone.utc) - timedelta(days=90)
            rows = await mdb.fetch_all(select([NodeCommit.oid])
                                       .where(and_(NodeCommit.oid.in_(stop_heads),
                                                   NodeCommit.committed_date > min_commit_time,
                                                   NodeCommit.acc_id.in_(meta_ids)))
                                       .order_by(desc(NodeCommit.committed_date))
                                       .limit(max_stop_heads))
            stop_heads = np.fromiter((r[0] for r in rows), dtype="U40", count=len(rows))
        first_parents = extract_first_parents(hashes, vertexes, edges, stop_heads, max_depth=1000)
        # We can still branch from an arbitrary point. Choose `max_partitions` graph partitions.
        if len(first_parents) >= max_inner_partitions:
            step = len(first_parents) // max_inner_partitions
            partition_seeds = first_parents[:max_inner_partitions * step:step]
        else:
            partition_seeds = first_parents
        partition_seeds = np.concatenate([stop_heads, partition_seeds])
        # the expansion factor is ~6x, so 2 * 25 -> 300
        stop_hashes = partition_dag(hashes, vertexes, edges, partition_seeds)
    else:
        stop_hashes = []
    batch_size = 20
    while len(head_hashes) > 0:
        new_edges = await _fetch_commit_history_edges(
            head_ids[:batch_size], stop_hashes, meta_ids, mdb)
        if not new_edges:
            new_edges = [(h, "", 0) for h in np.sort(np.unique(head_hashes[:batch_size]))]
        hashes, vertexes, edges = join_dags(hashes, vertexes, edges, new_edges)
        head_hashes = head_hashes[batch_size:]
        head_ids = head_ids[batch_size:]
        if len(head_hashes) > 0:
            collateral = np.where(
                hashes[searchsorted_inrange(hashes, head_hashes)] == head_hashes)[0]
            if len(collateral) > 0:
                head_hashes = np.delete(head_hashes, collateral)
                head_ids = np.delete(head_ids, collateral)
    return repo, hashes, vertexes, edges


async def _fetch_commit_history_edges(commit_ids: Iterable[str],
                                      stop_hashes: Iterable[str],
                                      meta_ids: Tuple[int, ...],
                                      mdb: databases.Database) -> List[Tuple]:
    # SQL credits: @dennwc
    quote = "`" if mdb.url.dialect == "sqlite" else ""
    if len(meta_ids) == 1:
        meta_id_sql = ("= %d" % meta_ids[0])
    else:
        meta_id_sql = "IN (%s)" % ", ".join(str(i) for i in meta_ids)
    query = f"""
    WITH RECURSIVE commit_history AS (
        SELECT
            p.child_id AS parent,
            p.{quote}index{quote} AS parent_index,
            pc.oid AS child_oid,
            cc.oid AS parent_oid,
            p.acc_id
        FROM
            github_node_commit_parents p
                LEFT JOIN github_node_commit pc ON p.parent_id = pc.id AND p.acc_id = pc.acc_id
                LEFT JOIN github_node_commit cc ON p.child_id = cc.id AND p.acc_id = cc.acc_id
        WHERE
            p.parent_id IN ('{"', '".join(commit_ids)}') AND p.acc_id {meta_id_sql}
        UNION
            SELECT
                p.child_id AS parent,
                p.{quote}index{quote} AS parent_index,
                pc.oid AS child_oid,
                cc.oid AS parent_oid,
                p.acc_id
            FROM
                github_node_commit_parents p
                    INNER JOIN commit_history h ON h.parent = p.parent_id AND p.acc_id = h.acc_id
                    LEFT JOIN github_node_commit pc ON p.parent_id = pc.id AND p.acc_id = pc.acc_id
                    LEFT JOIN github_node_commit cc ON p.child_id = cc.id AND p.acc_id = cc.acc_id
            WHERE     pc.oid NOT IN ('{"', '".join(stop_hashes)}')
                  AND p.parent_id NOT IN ('{"', '".join(commit_ids)}')
    ) SELECT
        child_oid,
        parent_oid,
        parent_index
    FROM
        commit_history;
    """
    async with mdb.connection() as conn:
        if isinstance(conn.raw_connection, asyncpg.connection.Connection):
            # this works much faster then iterate() / fetch_all()
            async with conn._query_lock:
                return await conn.raw_connection.fetch(query)
        else:
            return [tuple(r) for r in await conn.fetch_all(query)]
