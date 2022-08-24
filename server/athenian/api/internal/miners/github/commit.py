from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
import pickle
from typing import (
    Collection,
    Dict,
    Iterable,
    KeysView,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import aiomcache
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, desc, outerjoin, select, union_all
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached, middle_term_exptime, short_term_exptime
from athenian.api.db import (
    Database,
    DatabaseLike,
    add_pdb_hits,
    add_pdb_misses,
    dialect_specific_insert,
    ensure_db_datetime_tz,
)
from athenian.api.defer import defer
from athenian.api.internal.logical_repos import drop_logical_repo
from athenian.api.internal.miners.github.branches import BranchMiner, load_branch_commit_dates
from athenian.api.internal.miners.github.dag_accelerated import (
    append_missing_heads,
    extract_first_parents,
    extract_subdag,
    find_orphans,
    join_dags,
    partition_dag,
    searchsorted_inrange,
    verify_edges_integrity,
)
from athenian.api.internal.miners.types import DAG as DAGStruct
from athenian.api.internal.prefixer import Prefixer
from athenian.api.models.metadata.github import (
    Branch,
    NodeCommit,
    NodePullRequestCommit,
    PushCommit,
    Release,
    User,
)
from athenian.api.models.precomputed.models import GitHubCommitHistory
from athenian.api.tracing import sentry_span
from athenian.api.unordered_unique import in1d_str


class FilterCommitsProperty(Enum):
    """Primary commit filter modes."""

    NO_PR_MERGES = "no_pr_merges"
    BYPASSING_PRS = "bypassing_prs"


# hashes, vertex offsets in edges, edge indexes
DAG = Tuple[np.ndarray, np.ndarray, np.ndarray]


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda prop, date_from, date_to, repos, with_author, with_committer, only_default_branch, **kwargs: (  # noqa
        prop.value,
        date_from.timestamp(),
        date_to.timestamp(),
        ",".join(sorted(repos)),
        ",".join(sorted(with_author)) if with_author is not None and len(with_author) else "",
        ",".join(sorted(with_committer))
        if with_committer is not None and len(with_committer)
        else "",  # noqa
        "" if kwargs.get("columns") is None else ",".join(c.name for c in kwargs["columns"]),
        only_default_branch,
    ),
)
async def extract_commits(
    prop: FilterCommitsProperty,
    date_from: datetime,
    date_to: datetime,
    repos: Collection[str],
    with_author: Optional[Collection[str]],
    with_committer: Optional[Collection[str]],
    only_default_branch: bool,
    branch_miner: Optional[BranchMiner],
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: DatabaseLike,
    pdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
    columns: Optional[List[InstrumentedAttribute]] = None,
) -> pd.DataFrame:
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
    if with_author is not None and len(with_author):
        user_logins.update(with_author)
    if with_committer is not None and len(with_committer):
        user_logins.update(with_committer)
    if user_logins:
        user_ids = dict(
            await mdb.fetch_all(
                select([User.login, User.node_id]).where(
                    and_(User.login.in_(user_logins), User.acc_id.in_(meta_ids)),
                ),
            ),
        )
        del user_logins
    else:
        user_ids = {}
    if with_author is not None and len(with_author):
        author_ids = []
        for u in with_author:
            try:
                author_ids.append(user_ids[u])
            except KeyError:
                continue
        sql_filters.append(PushCommit.author_user_id.in_(author_ids))
    if with_committer is not None and len(with_committer):
        committer_ids = []
        for u in with_committer:
            try:
                committer_ids.append(user_ids[u])
            except KeyError:
                continue
        sql_filters.append(PushCommit.committer_user_id.in_(committer_ids))
    if columns is None:
        cols_query, cols_df = [PushCommit], PushCommit
    else:
        for col in (PushCommit.node_id, PushCommit.repository_full_name, PushCommit.sha):
            if col not in columns:
                columns.append(col)
        cols_query = cols_df = columns
    if prop == FilterCommitsProperty.NO_PR_MERGES:
        commits_task = read_sql_query(select(cols_query).where(and_(*sql_filters)), mdb, cols_df)
    elif prop == FilterCommitsProperty.BYPASSING_PRS:
        commits_task = read_sql_query(
            select(cols_query)
            .select_from(
                outerjoin(
                    PushCommit,
                    NodePullRequestCommit,
                    and_(
                        PushCommit.node_id == NodePullRequestCommit.commit_id,
                        PushCommit.acc_id == NodePullRequestCommit.acc_id,
                    ),
                ),
            )
            .where(and_(NodePullRequestCommit.commit_id.is_(None), *sql_filters)),
            mdb,
            cols_df,
        )
    else:
        raise AssertionError('Unsupported primary commit filter "%s"' % prop)
    tasks = [
        commits_task,
        fetch_repository_commits_from_scratch(
            repos,
            branch_miner,
            only_default_branch,
            True,
            prefixer,
            account,
            meta_ids,
            mdb,
            pdb,
            cache,
        ),
    ]
    commits, (dags, branches, default_branches) = await gather(*tasks, op="extract_commits/fetch")
    candidates_count = len(commits)
    if only_default_branch:
        commits = _take_commits_in_default_branches(commits, dags, branches, default_branches)
        log.info("Removed side branch commits: %d / %d", len(commits), candidates_count)
    else:
        commits = _remove_force_push_dropped(commits, dags)
        log.info("Removed force push dropped commits: %d / %d", len(commits), candidates_count)
    for number_prop in (PushCommit.additions, PushCommit.deletions, PushCommit.changed_files):
        try:
            number_col = commits[number_prop.name]
        except KeyError:
            continue
        nans = commits[PushCommit.node_id.name].take(np.where(number_col.isna())[0])
        if not nans.empty:
            log.error("[DEV-546] Commits have NULL in %s: %s", number_prop.name, ", ".join(nans))
    return commits


def _take_commits_in_default_branches(
    commits: pd.DataFrame,
    dags: Dict[str, Tuple[bool, DAG]],
    branches: pd.DataFrame,
    default_branches: Dict[str, str],
) -> pd.DataFrame:
    if commits.empty:
        return commits
    branch_repos = branches[Branch.repository_full_name.name].values.astype("U")
    branch_names = branches[Branch.branch_name.name].values.astype("U")
    branch_repos_names = np.char.add(np.char.add(branch_repos, "|"), branch_names)
    default_repos_names = np.array([f"{k}|{v}" for k, v in default_branches.items()], dtype="U")
    default_mask = np.in1d(branch_repos_names, default_repos_names, assume_unique=True)
    branch_repos = branch_repos[default_mask]
    branch_hashes = branches[Branch.commit_sha.name].values[default_mask]
    repos_order = np.argsort(branch_repos)
    branch_repos = branch_repos[repos_order]
    branch_hashes = branch_hashes[repos_order]

    commit_repos, commit_repo_indexes = np.unique(
        commits[PushCommit.repository_full_name.name].values.astype("U"), return_inverse=True,
    )
    commit_repos_in_branches_mask = np.in1d(commit_repos, branch_repos, assume_unique=True)
    branch_repos_in_commits_mask = np.in1d(branch_repos, commit_repos, assume_unique=True)
    branch_repos = branch_repos[branch_repos_in_commits_mask]
    branch_hashes = branch_hashes[branch_repos_in_commits_mask]
    commit_hashes = commits[PushCommit.sha.name].values

    accessible_indexes = []
    for repo, head_sha, commit_repo_index in zip(
        branch_repos, branch_hashes, np.nonzero(commit_repos_in_branches_mask)[0],
    ):
        repo_indexes = np.nonzero(commit_repo_indexes == commit_repo_index)[0]
        repo_hashes = commit_hashes[repo_indexes]
        default_branch_hashes = extract_subdag(*dags[repo][1], np.array([head_sha]))[0]
        accessible_indexes.append(
            repo_indexes[np.in1d(repo_hashes, default_branch_hashes, assume_unique=True)],
        )
    if accessible_indexes:
        accessible_indexes = np.sort(np.concatenate(accessible_indexes))
    return commits.take(accessible_indexes)


def _remove_force_push_dropped(
    commits: pd.DataFrame,
    dags: Dict[str, Tuple[bool, DAG]],
) -> pd.DataFrame:
    if commits.empty:
        return commits
    repos_order, indexes = np.unique(
        commits[PushCommit.repository_full_name.name].values, return_inverse=True,
    )
    hashes = commits[PushCommit.sha.name].values
    accessible_indexes = []
    for i, repo in enumerate(repos_order):
        repo_indexes = np.flatnonzero(indexes == i)
        repo_hashes = hashes[repo_indexes]
        accessible_indexes.append(
            repo_indexes[np.in1d(repo_hashes, dags[repo][1][0], assume_unique=True)],
        )
    accessible_indexes = np.sort(np.concatenate(accessible_indexes))
    return commits.take(accessible_indexes)


@sentry_span
@cached(
    exptime=middle_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, branches, columns, prune, **_: (
        ",".join(sorted(repos)),
        b",".join(
            np.sort(
                branches[columns[0] if isinstance(columns[0], str) else columns[0].name].values,
            ),
        ),
        prune,
    )
    if not branches.empty
    else None,
    refresh_on_access=True,
)
async def fetch_repository_commits(
    repos: Dict[str, Tuple[bool, DAG]],
    branches: pd.DataFrame,
    columns: Tuple[
        Union[str, InstrumentedAttribute],
        Union[str, InstrumentedAttribute],
        Union[str, InstrumentedAttribute],
        Union[str, InstrumentedAttribute],
    ],
    prune: bool,
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> Dict[str, Tuple[bool, DAG]]:
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
    :return: Map from repository names to their DAG consistency indicators and bodies.
    """
    if branches.empty:
        if not prune:
            return repos
        return {key: (True, _empty_dag()) for key in repos}
    missed_counter = 0
    repo_heads = {}
    sha_col, id_col, dt_col, repo_col = (c if isinstance(c, str) else c.name for c in columns)
    result = {}
    tasks = []
    df_shas = branches[sha_col].values
    sha_order = np.argsort(df_shas)
    df_shas = df_shas[sha_order]
    df_ids = branches[id_col].values[sha_order]
    df_dts = branches[dt_col].values[sha_order]
    df_repos = branches[repo_col].values[sha_order]
    del branches
    unique_repos, index_map, counts = np.unique(df_repos, return_inverse=True, return_counts=True)
    repo_order = np.argsort(index_map)
    offsets = np.zeros(len(counts) + 1, dtype=int)
    np.cumsum(counts, out=offsets[1:])
    for i, repo in enumerate(unique_repos):
        required_shas = df_shas[repo_order[offsets[i] : offsets[i + 1]]]
        required_shas = required_shas[required_shas != b""]
        repo_heads[repo] = required_shas
        try:
            _, (hashes, vertexes, edges) = repos[drop_logical_repo(repo)]
        except KeyError:
            # totally OK, `branches` may include repositories from other ForSet-s
            continue
        if len(hashes) > 0:
            found_indexes = searchsorted_inrange(hashes, required_shas)
            missed_mask = hashes[found_indexes] != required_shas
            missed_counter += missed_mask.sum()
            missed_shas = required_shas[missed_mask]  # these hashes do not exist in the p-DAG
        else:
            missed_shas = required_shas
        if len(missed_shas) > 0:
            missed_indexes = np.searchsorted(df_shas, missed_shas)
            # heuristic: order the heads from most to least recent
            order = np.argsort(df_dts[missed_indexes])[::-1]
            missed_shas = missed_shas[order]
            missed_ids = df_ids[missed_indexes[order]]
            tasks.append(
                _fetch_commit_history_dag(
                    hashes, vertexes, edges, missed_shas, missed_ids, repo, meta_ids, mdb,
                ),
            )
        else:
            if prune:
                hashes, vertexes, edges = extract_subdag(hashes, vertexes, edges, required_shas)
            result[repo] = True, (hashes, vertexes, edges)
    # traverse commits starting from the missing branch heads
    add_pdb_hits(pdb, "fetch_repository_commits", len(df_shas) - missed_counter)
    add_pdb_misses(pdb, "fetch_repository_commits", missed_counter)
    if tasks:
        new_dags = await gather(*tasks, op="fetch_repository_commits/mdb")
        sql_values = []
        for consistent, repo, hashes, vertexes, edges in new_dags:
            assert (hashes[1:] > hashes[:-1]).all(), repo
            sql_values.append(
                GitHubCommitHistory(
                    acc_id=account,
                    repository_full_name=repo,
                    dag=DAGStruct.from_fields(hashes=hashes, vertexes=vertexes, edges=edges).data,
                )
                .create_defaults()
                .explode(with_primary_keys=True),
            )
            if prune:
                hashes, vertexes, edges = extract_subdag(hashes, vertexes, edges, repo_heads[repo])
            result[repo] = consistent, (hashes, vertexes, edges)
        sql = (await dialect_specific_insert(pdb))(GitHubCommitHistory)
        sql = sql.on_conflict_do_update(
            index_elements=GitHubCommitHistory.__table__.primary_key.columns,
            set_={
                GitHubCommitHistory.dag.name: sql.excluded.dag,
                GitHubCommitHistory.updated_at.name: sql.excluded.updated_at,
            },
        )

        async def execute():
            if pdb.url.dialect == "sqlite":
                async with pdb.connection() as pdb_conn:
                    async with pdb_conn.transaction():
                        await pdb_conn.execute_many(sql, sql_values)
            else:
                # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
                await pdb.execute_many(sql, sql_values)

        await defer(execute(), "fetch_repository_commits/pdb")
    for repo, pdag in repos.items():
        if repo not in result:
            result[repo] = (True, _empty_dag()) if prune else pdag
    return result


BRANCH_FETCH_COMMITS_COLUMNS = (
    Branch.commit_sha,
    Branch.commit_id,
    Branch.commit_date,
    Branch.repository_full_name,
)
RELEASE_FETCH_COMMITS_COLUMNS = (
    Release.sha,
    Release.commit_id,
    Release.published_at,
    Release.repository_full_name,
)
COMMIT_FETCH_COMMITS_COLUMNS = (
    PushCommit.sha,
    PushCommit.node_id,
    PushCommit.committed_date,
    PushCommit.repository_full_name,
)


@sentry_span
async def fetch_repository_commits_no_branch_dates(
    repos: Dict[str, Tuple[bool, DAG]],
    branches: pd.DataFrame,
    columns: Tuple[str, str, str, str],
    prune: bool,
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> Dict[str, Tuple[bool, DAG]]:
    """
    Load full commit DAGs for the given repositories.

    The difference with fetch_repository_commits is that `branches` may possibly miss the commit \
    dates. If that is the case, we fetch the commit dates.
    """
    await load_branch_commit_dates(branches, meta_ids, mdb)
    return await fetch_repository_commits(
        repos, branches, columns, prune, account, meta_ids, mdb, pdb, cache,
    )


@sentry_span
async def fetch_repository_commits_from_scratch(
    repos: Iterable[str],
    branch_miner: BranchMiner,
    only_default_branch: bool,
    prune: bool,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> Tuple[Dict[str, Tuple[bool, DAG]], pd.DataFrame, Dict[str, str]]:
    """
    Load full commit DAGs for the given repositories.

    The difference with fetch_repository_commits is that we don't have `branches`. We load them
    in-place.
    """
    (branches, defaults), pdags = await gather(
        branch_miner.extract_branches(repos, prefixer, meta_ids, mdb, cache),
        fetch_precomputed_commit_history_dags(repos, account, pdb, cache),
    )
    if only_default_branch:
        branch_index = np.char.add(
            branches[Branch.repository_full_name.name].values.astype("U"),
            branches[Branch.branch_name.name].values.astype("U"),
        )
        if isinstance(repos, (set, frozenset, KeysView)):
            repos = list(repos)
        default_set = np.char.add(
            np.array(repos, dtype="U"),
            np.array([defaults.get(r) for r in repos], dtype="U"),
        )
        branches = branches.take(np.flatnonzero(in1d_str(branch_index, default_set)))
    dags = await fetch_repository_commits_no_branch_dates(
        pdags, branches, BRANCH_FETCH_COMMITS_COLUMNS, prune, account, meta_ids, mdb, pdb, cache,
    )
    return dags, branches, defaults


@sentry_span
@cached(
    exptime=middle_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, **_: (",".join(sorted(repos)),),
)
async def fetch_precomputed_commit_history_dags(
    repos: Iterable[str],
    account: int,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> Dict[str, Tuple[bool, DAG]]:
    """Load commit DAGs from the pdb."""
    ghrc = GitHubCommitHistory
    format_version = ghrc.__table__.columns[ghrc.format_version.key].default.arg
    with sentry_sdk.start_span(op="fetch_precomputed_commit_history_dags/pdb"):
        rows = await pdb.fetch_all(
            select([ghrc.repository_full_name, ghrc.dag]).where(
                and_(
                    ghrc.format_version == format_version,
                    ghrc.repository_full_name.in_(repos),
                    ghrc.acc_id == account,
                ),
            ),
        )
    dags = {
        row[ghrc.repository_full_name.name]: (
            True,
            ((dag := DAGStruct(row[ghrc.dag.name])).hashes, dag.vertexes, dag.edges),
        )
        for row in rows
    }
    for repo in repos:
        if repo not in dags:
            dags[repo] = True, _empty_dag()
    return dags


def _empty_dag() -> DAG:
    return np.array([], dtype="S40"), np.array([0], dtype=np.uint32), np.array([], dtype=np.uint32)


@sentry_span
async def _fetch_commit_history_dag(
    hashes: np.ndarray,
    vertexes: np.ndarray,
    edges: np.ndarray,
    head_hashes: Sequence[Union[str, bytes]],
    head_ids: Sequence[int],
    repo: str,
    meta_ids: Tuple[int, ...],
    mdb: Database,
) -> Tuple[bool, str, np.ndarray, np.ndarray, np.ndarray]:
    consistent = True
    max_stop_heads = 25
    max_inner_partitions = 25
    log = logging.getLogger("%s._fetch_commit_history_dag" % metadata.__package__)
    # there can be duplicates, remove them
    head_hashes = np.asarray(head_hashes, dtype="S40")
    head_ids = np.asarray(head_ids, dtype=int)
    _, unique_indexes = np.unique(head_hashes, return_index=True)
    head_hashes = head_hashes[unique_indexes]
    head_ids = head_ids[unique_indexes]
    # find max `max_stop_heads` top-level most recent commit hashes
    stop_heads = hashes[np.delete(np.arange(len(hashes)), np.unique(edges))]
    if len(stop_heads) > 0:
        if len(stop_heads) > max_stop_heads:
            min_commit_time = datetime.now(timezone.utc) - timedelta(days=90)
            rows = await mdb.fetch_all(
                select([NodeCommit.oid])
                .where(
                    and_(
                        NodeCommit.oid.in_(stop_heads),
                        NodeCommit.committed_date > min_commit_time,
                        NodeCommit.acc_id.in_(meta_ids),
                    ),
                )
                .order_by(desc(NodeCommit.committed_date))
                .limit(max_stop_heads),
            )
            stop_heads = np.fromiter((r[0] for r in rows), dtype="S40", count=len(rows))
        first_parents = extract_first_parents(hashes, vertexes, edges, stop_heads, max_depth=1000)
        # We can still branch from an arbitrary point. Choose `max_partitions` graph partitions.
        if len(first_parents) >= max_inner_partitions:
            step = len(first_parents) // max_inner_partitions
            partition_seeds = first_parents[: max_inner_partitions * step : step]
        else:
            partition_seeds = first_parents
        partition_seeds = np.concatenate([stop_heads, partition_seeds])
        assert partition_seeds.dtype.char == "S"
        # the expansion factor is ~6x, so 2 * 25 -> 300
        with sentry_sdk.start_span(
            op="partition_dag", description="%d %d" % (len(hashes), len(partition_seeds)),
        ):
            stop_hashes = partition_dag(hashes, vertexes, edges, partition_seeds)
    else:
        stop_hashes = []
    batch_size = 20
    while len(head_hashes) > 0:
        new_edges = await _fetch_commit_history_edges(
            head_ids[:batch_size], stop_hashes, meta_ids, mdb,
        )
        if bads := verify_edges_integrity(new_edges):
            log.warning(
                "some new DAG edges are not consistent: %s",
                [new_edges[i] for i in bads],
            )
            consistent = False
            for i in bads[::-1]:
                new_edges.pop(i)
        append_missing_heads(new_edges, head_hashes[:batch_size])
        if orphans := find_orphans(new_edges, hashes):
            committed_dates = dict(
                await mdb.fetch_all(
                    select(NodeCommit.oid, NodeCommit.committed_date).where(
                        and_(NodeCommit.acc_id.in_(meta_ids), NodeCommit.oid.in_(orphans)),
                    ),
                ),
            )
            removed_orphans = set()
            for leaf, indexes in orphans.items():
                try:
                    committed_date = committed_dates[leaf]
                except KeyError:
                    log.error("failed to fetch committed_date of %s", leaf)
                    removed_orphans.update(indexes)
                else:
                    committed_date = ensure_db_datetime_tz(committed_date, mdb)
                    if datetime.now(timezone.utc) - committed_date < timedelta(days=1, hours=6):
                        log.warning("skipping an orphan which is suspiciously young: %s", leaf)
                        removed_orphans.update(indexes)
            if removed_orphans:
                consistent = False
                for i in sorted(removed_orphans, reverse=True):
                    new_edges.pop(i)
        hashes, vertexes, edges = join_dags(hashes, vertexes, edges, new_edges)
        head_hashes = head_hashes[batch_size:]
        head_ids = head_ids[batch_size:]
        if len(head_hashes) > 0 and len(hashes) > 0:
            collateral = np.flatnonzero(
                hashes[searchsorted_inrange(hashes, head_hashes)] == head_hashes,
            )
            if len(collateral) > 0:
                head_hashes = np.delete(head_hashes, collateral)
                head_ids = np.delete(head_ids, collateral)
    return consistent, repo, hashes, vertexes, edges


async def _fetch_commit_history_edges(
    commit_ids: Iterable[int],
    stop_hashes: Iterable[bytes],
    meta_ids: Tuple[int, ...],
    mdb: Database,
) -> List[Tuple]:
    """
    Query metadata DB for the new commit DAG edges.

    We recursively traverse github.node_commit_edge_parents starting from `commit_ids`.
    Initial SQL credits: @dennwc.

    We return nodes in the native DB order, that's the opposite of Git's parent-child.
    `stop_hashes` are the recursion terminators so that we don't traverse the full DAG every time.

    We don't include the edges from the outside to the first parents (`commit_ids`). This means
    that if some of `commit_ids` do not have children, there will be 0 edges with them.
    """
    assert isinstance(mdb, Database), "fetch_all() must be patched to avoid re-wrapping"
    if mdb.url.dialect == "sqlite":
        rq = "`"
        tq = '"'
    else:
        rq = tq = ""
    if len(meta_ids) == 1:
        meta_id_sql = "= %d" % meta_ids[0]
    else:
        meta_id_sql = "IN (%s)" % ", ".join(str(i) for i in meta_ids)
    query = f"""
    WITH RECURSIVE commit_history AS (
        SELECT
            p.child_id,
            p.{rq}index{rq} AS parent_index,
            pc.oid AS parent_oid,
            cc.oid AS child_oid,
            p.acc_id
        FROM
            {tq}github.node_commit_edge_parents{tq} p
                LEFT JOIN {tq}github.node_commit{tq} pc ON p.parent_id = pc.graph_id AND p.acc_id = pc.acc_id
                LEFT JOIN {tq}github.node_commit{tq} cc ON p.child_id = cc.graph_id AND p.acc_id = cc.acc_id
        WHERE
            p.parent_id IN ({", ".join(map(str, commit_ids))}) AND p.acc_id {meta_id_sql}
        UNION
            SELECT
                p.child_id,
                p.{rq}index{rq} AS parent_index,
                h.child_oid AS parent_oid,
                cc.oid AS child_oid,
                p.acc_id
            FROM
                {tq}github.node_commit_edge_parents{tq} p
                    INNER JOIN commit_history h ON p.parent_id = h.child_id AND p.acc_id = h.acc_id
                    LEFT JOIN {tq}github.node_commit{tq} cc ON p.child_id = cc.graph_id AND p.acc_id = cc.acc_id
            WHERE h.child_oid NOT IN ('{"', '".join(h.decode() for h in stop_hashes)}')
    ) SELECT
        parent_oid,
        child_oid,
        parent_index
    FROM
        commit_history
    """  # noqa
    rows = await mdb.fetch_all(query)
    if mdb.url.dialect == "sqlite":
        rows = [tuple(r) for r in rows]
    return rows


@sentry_span
async def fetch_dags_with_commits(
    commits: Mapping[str, Sequence[int]],
    prune: bool,
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> Tuple[Dict[str, Tuple[bool, DAG]], pd.DataFrame]:
    """
    Load full commit DAGs for the given commit node IDs mapped from repository names.

    :param commits: repository name -> sequence of commit node IDs.
    :return: 1. DAGs that contain the specified commits, by repository name. \
             2. DataFrame with loaded `commits` - *not with all the commits in the DAGs*.
    """
    commits, pdags = await gather(
        _fetch_commits_for_dags(commits, meta_ids, mdb, cache),
        fetch_precomputed_commit_history_dags(commits, account, pdb, cache),
        op="fetch_dags_with_commits/prepare",
    )
    dags = await fetch_repository_commits(
        pdags, commits, COMMIT_FETCH_COMMITS_COLUMNS, prune, account, meta_ids, mdb, pdb, cache,
    )
    return dags, commits


@sentry_span
@cached(
    exptime=middle_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda commits, **_: (
        ";".join(
            "%s: %s" % (k, ",".join(map(str, sorted(v)))) for k, v in sorted(commits.items())
        ),
    ),
    refresh_on_access=True,
)
async def _fetch_commits_for_dags(
    commits: Mapping[str, Sequence[int]],
    meta_ids: Tuple[int, ...],
    mdb: Database,
    cache: Optional[aiomcache.Client],
) -> pd.DataFrame:
    if not commits:
        return pd.DataFrame()
    queries = [
        select(COMMIT_FETCH_COMMITS_COLUMNS).where(
            and_(
                PushCommit.repository_full_name == repo,
                PushCommit.acc_id.in_(meta_ids),
                PushCommit.node_id.in_any_values(nodes),
            ),
        )
        for repo, nodes in commits.items()
    ]
    return await read_sql_query(union_all(*queries), mdb, COMMIT_FETCH_COMMITS_COLUMNS)
