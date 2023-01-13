from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from typing import Collection, Iterable, Iterator, Optional

import aiomcache
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import func, select

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import CancelCache, cached, cached_methods, short_term_exptime
from athenian.api.db import DatabaseLike, dialect_specific_insert
from athenian.api.defer import defer
from athenian.api.internal.logical_repos import coerce_logical_repos
from athenian.api.internal.prefixer import Prefixer
from athenian.api.models.metadata.github import Branch, NodeRepositoryRef
from athenian.api.models.persistentdata.models import HealthMetric
from athenian.api.pandas_io import deserialize_args, serialize_args
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import numpy_struct
from athenian.precomputer.db.models import GitHubBranches


@numpy_struct
class PrecomputedBranches:
    """Branch metadata aggregated by repo."""

    class Immutable:
        """
        Immutable fields repeat Branch columns.

        We generate `dtype` from this spec.

        The format is all branches per repo, hence arrays.
        """

        branch_id: [int]
        branch_name: [str]
        is_default: [bool]
        commit_id: [int]
        commit_sha: ["S40"]  # noqa: F821
        commit_date: ["datetime64[s]"]  # noqa: F821

    class Optional:
        """Mutable fields that are None by default. We do not serialize them."""

        repository_full_name: str
        repository_node_id: int


@dataclass(slots=True)
class BranchMinerMetrics:
    """Branch source data error statistics."""

    count: int
    empty_count: int
    no_default: int

    @classmethod
    def empty(cls) -> "BranchMinerMetrics":
        """Initialize a new BranchMinerMetrics instance filled with zeros."""
        return BranchMinerMetrics(0, 0, 0)

    def as_db(self) -> Iterator[HealthMetric]:
        """Generate HealthMetric-s from this instance."""
        yield HealthMetric(name="branches_count", value=self.count)
        yield HealthMetric(name="branches_empty_count", value=self.empty_count)
        yield HealthMetric(name="branches_no_default", value=self.no_default)


@cached_methods
class BranchMiner:
    """Load information related to branches."""

    @classmethod
    async def load_branches(
        cls,
        repos: Optional[Iterable[str]],
        prefixer: Prefixer,
        account: int,
        meta_ids: tuple[int, ...],
        mdb: DatabaseLike,
        pdb: Optional[DatabaseLike],
        cache: Optional[aiomcache.Client],
        strip: bool = False,
        fresh: bool = True,  # wip, let pdb fill up
        metrics: Optional[BranchMinerMetrics] = None,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        Fetch branches in the given repositories and extract the default branch names.

        :param strip: Value indicating whether the repository names are prefixed.
        :param fresh: Value indicating whether we must bypass pdb and load branches from scratch.
        :param metrics: Report error statistics by mutating this object.
        """
        return (
            await cls._load_branches(
                repos, prefixer, account, meta_ids, mdb, pdb, cache, strip, fresh, metrics,
            )
        )[:-1]

    def _postprocess_branches(
        result: tuple[pd.DataFrame, dict[str, str], bool],
        repos: Optional[Iterable[str]],
        strip: bool,
        **_,
    ) -> tuple[pd.DataFrame, dict[str, str], bool]:
        branches, default_branches, with_repos = result
        if repos is None:
            if with_repos:
                raise CancelCache()
            return branches, default_branches, False
        if strip:
            repos = [r.split("/", 1)[1] for r in repos]
        try:
            default_branches = {k: default_branches[k] for k in repos}
        except KeyError as e:
            raise CancelCache() from e
        # it should be okay to potentially return some extra branches in the dataframe
        return branches, default_branches, True

    @classmethod
    @sentry_span
    @cached(
        exptime=short_term_exptime,
        serialize=serialize_args,
        deserialize=deserialize_args,
        key=lambda account, **_: (account,),
        postprocess=_postprocess_branches,
    )
    async def _load_branches(
        cls,
        repos: Optional[Iterable[str]],
        prefixer: Prefixer,
        account: int,
        meta_ids: tuple[int, ...],
        mdb: DatabaseLike,
        pdb: DatabaseLike,
        cache: Optional[aiomcache.Client],
        strip: bool,
        fresh: bool,
        metrics: Optional[BranchMinerMetrics],
    ) -> tuple[pd.DataFrame, dict[str, str], bool]:
        if strip and repos is not None:
            repos = [r.split("/", 1)[1] for r in repos]
        if repos is not None:
            repos = coerce_logical_repos(repos)
            repo_ids = [prefixer.repo_name_to_node.get(r) for r in repos]
        else:
            repo_ids = None
        if fresh:
            branches = await cls._fetch_branches(repo_ids, None, prefixer, meta_ids, mdb)
            if pdb is not None:
                await defer(
                    cls._store_precomputed_branches(account, branches, repo_ids, pdb),
                    f"store_branches({len(branches)})",
                )
        else:
            new_branches, old_branches = await gather(
                cls._fetch_branches(
                    repo_ids,
                    datetime.now(timezone.utc) - timedelta(hours=2),
                    prefixer,
                    meta_ids,
                    mdb,
                ),
                cls._fetch_precomputed_branches(account, repo_ids, prefixer, pdb),
            )
            if old_branches.empty:
                branches = new_branches
            elif new_branches.empty:
                branches = old_branches
            else:
                old_indexes = np.flatnonzero(
                    np.in1d(
                        old_branches[Branch.branch_id.name].values,
                        new_branches[Branch.branch_id.name].values,
                        invert=True,
                    ),
                )
                if len(old_indexes):
                    if len(old_indexes) < len(old_branches):
                        old_branches = old_branches.take(old_indexes)
                    branches = pd.concat([new_branches, old_branches], ignore_index=True)
                else:
                    branches = new_branches
            if not new_branches.empty:
                await defer(
                    cls._store_precomputed_branches(
                        account,
                        branches,
                        new_branches[Branch.repository_node_id.name].unique(),
                        pdb,
                    ),
                    f"store_branches({len(branches)})",
                )
        log = logging.getLogger("%s.extract_default_branches" % metadata.__package__)
        default_branches = {}
        ambiguous_defaults = {}
        branch_repos = branches[Branch.repository_full_name.name].values
        unique_branch_repos, index_map, counts = np.unique(
            branch_repos, return_inverse=True, return_counts=True,
        )
        if repos is None:
            repos = unique_branch_repos
        order = np.argsort(index_map)
        branch_names = branches[Branch.branch_name.name].values[order]
        branch_defaults = branches[Branch.is_default.name].values[order]
        branch_commit_ids = branches[Branch.commit_id.name].values[order]
        pos = 0
        for i, repo in enumerate(unique_branch_repos):
            next_pos = pos + counts[i]
            default_indexes = np.flatnonzero(branch_defaults[pos:next_pos])
            repo_branch_names = branch_names[pos:next_pos]
            if len(default_indexes) > 0:
                default_branch = repo_branch_names[default_indexes[0]]
            else:
                if "master" in repo_branch_names:
                    default_branch = "master"
                elif "main" in repo_branch_names:
                    default_branch = "main"
                elif len(repo_branch_names) == 1:
                    default_branch = repo_branch_names[0]
                elif len(repo_branch_names) > 0:
                    ambiguous_defaults[repo] = (branch_commit_ids[pos:next_pos], repo_branch_names)
                    default_branch = "<ambiguous>"
                else:
                    default_branch = "master"
                log.warning(
                    "%s does not have an explicit default branch among %d listed, set to %s",
                    repo,
                    len(repo_branch_names),
                    default_branch,
                )
                if metrics is not None:
                    metrics.no_default += 1
            default_branches[repo] = default_branch
            pos = next_pos
        if ambiguous_defaults:
            map_indexes = np.flatnonzero(
                np.in1d(
                    branches[Branch.commit_id.name].values,
                    np.concatenate([rb[0] for rb in ambiguous_defaults.values()]),
                ),
            )
            committed_dates = dict(
                zip(
                    branches[Branch.commit_id.name].values[map_indexes],
                    branches[Branch.commit_date.name].values[map_indexes],
                ),
            )
            for repo, (repo_commit_ids, repo_branch_names) in ambiguous_defaults.items():
                default_branch = max_date = None
                for name, commit_id in zip(repo_branch_names, repo_commit_ids):
                    if (commit_date := committed_dates.get(commit_id)) is None:
                        continue
                    if max_date is None or max_date < commit_date:
                        max_date = commit_date
                        default_branch = name
                if default_branch is None:
                    default_branch = "master"
                log.warning(
                    "resolved <ambiguous> default branch in %s to %s @ %s",
                    repo,
                    default_branch,
                    max_date,
                )
                default_branches[repo] = default_branch
        if zero_branch_repos := {repo for repo in repos if repo not in default_branches}:
            deleted_repos = zero_branch_repos - prefixer.repo_name_to_node.keys()
            if deleted_repos:
                for repo in deleted_repos:
                    default_branches[repo] = "master"
                log.error("some repositories do not exist: %s", deleted_repos)
            if zero_names := (zero_branch_repos & prefixer.repo_name_to_node.keys()):
                zero_nodes = [prefixer.repo_name_to_node[k] for k in zero_names]
                rows = await mdb.fetch_all(
                    select(
                        NodeRepositoryRef.parent_id,
                        func.count(NodeRepositoryRef.child_id).label("numrefs"),
                    )
                    .where(
                        NodeRepositoryRef.acc_id.in_(meta_ids),
                        NodeRepositoryRef.parent_id.in_(zero_nodes),
                    )
                    .group_by(NodeRepositoryRef.parent_id),
                )
                refs = {r[NodeRepositoryRef.parent_id.name]: r["numrefs"] for r in rows}
                reported_repos = set()
                warnings = []
                errors = []
                for node_id, full_name in zip(zero_nodes, zero_names):
                    if full_name not in reported_repos:
                        (warnings if refs.get(node_id, 0) == 0 else errors).append(full_name)
                        default_branches[full_name] = "master"
                        reported_repos.add(full_name)
                for report, items in ((log.warning, warnings), (log.error, errors)):
                    if items:
                        report("the following repositories have 0 branches: %s", items)
                        if metrics is not None:
                            metrics.empty_count += len(items)
        if metrics is not None:
            metrics.count = len(branches)
        return branches, default_branches, repo_ids is not None

    _postprocess_branches = staticmethod(_postprocess_branches)

    @classmethod
    async def reset_cache(cls, account: int, cache: Optional[aiomcache.Client]) -> None:
        """Drop the cache of load_branches()."""
        await gather(
            *(
                cls._load_branches.reset_cache(
                    # cached method is unbound, add one more param for cls
                    cls,
                    None,
                    None,
                    account,
                    (),
                    None,
                    None,
                    cache,
                    strip,
                    False,
                    None,
                )
                for strip in (False, True)
            ),
        )

    @classmethod
    @sentry_span
    async def _fetch_branches(
        cls,
        repos: Optional[Collection[int]],
        since: Optional[datetime],
        prefixer: Prefixer,
        meta_ids: tuple[int, ...],
        mdb: DatabaseLike,
    ) -> pd.DataFrame:
        columns = [
            Branch.branch_id,
            Branch.branch_name,
            Branch.repository_node_id,
            Branch.repository_full_name,
            Branch.is_default,
            Branch.commit_id,
            Branch.commit_sha,
            Branch.commit_date,
        ]
        query_repo_col = (
            Branch.repository_node_id if since is None else Branch.commit_repository_node_id
        )
        query = select(*columns).where(
            *((query_repo_col.in_(repos),) if repos is not None else ()),
            Branch.acc_id.in_(meta_ids),
            *((Branch.commit_date >= since,) if since is not None else ()),
        )
        if since is None:
            if repos is not None:
                scan = (
                    "IndexScan"
                    if len(repos) < len(prefixer.repo_node_to_name) * 0.8
                    else "BitmapScan"
                )
                query = query.with_statement_hint(f"{scan}(ref node_ref_heads_repository_id)")
            query = (
                query.with_statement_hint("IndexOnlyScan(c node_commit_repository_target)")
                .with_statement_hint(
                    "Rows(ref c repo rr n1 n2 "
                    f"*{len(repos) if repos is not None else len(prefixer.repo_node_to_name)})",
                )
                .with_statement_hint("Rows(ref c repo *100)")
                .with_statement_hint("Rows(ref c repo rr *100)")
                .with_statement_hint("set(join_collapse_limit 1)")
            )
        else:
            # Postgres still thinks we will fetch 0 rows, but that's OK on a narrow time window
            query = query.with_statement_hint("IndexOnlyScan(c github_node_commit_check_runs)")
        return await read_sql_query(query, mdb, columns)

    @classmethod
    @sentry_span
    async def _fetch_precomputed_branches(
        cls,
        account: int,
        repos: Optional[Collection[int]],
        prefixer: Prefixer,
        pdb: DatabaseLike,
    ) -> pd.DataFrame:
        rows = await pdb.fetch_all(
            select(GitHubBranches.repository_node_id, GitHubBranches.data).where(
                GitHubBranches.acc_id == account,
                GitHubBranches.format_version
                == GitHubBranches.__table__.columns[GitHubBranches.format_version.key].default.arg,
                *((GitHubBranches.repository_node_id.in_(repos),) if repos is not None else ()),
            ),
        )
        if not rows:
            return pd.DataFrame()
        repo_node_to_name = prefixer.repo_node_to_name.__getitem__
        branches = [
            PrecomputedBranches(
                r[1], repository_node_id=r[0], repository_full_name=repo_node_to_name(r[0]),
            )
            for r in rows
        ]
        columns = {}
        for col in PrecomputedBranches.dtype.names:
            try:
                values, borders = PrecomputedBranches.vectorize_field(branches, col)
            except NotImplementedError:
                values = np.concatenate([rb[col] for rb in branches])
            if col == Branch.commit_date.name:
                values = pd.Series(values, dtype=pd.DatetimeTZDtype(tz=timezone.utc))
            columns[col] = values
        lengths = np.diff(borders)
        columns[Branch.repository_node_id.name] = np.repeat(
            [b.repository_node_id for b in branches], lengths,
        )
        columns[Branch.repository_full_name.name] = np.repeat(
            [b.repository_full_name for b in branches], lengths,
        )
        return pd.DataFrame(columns)

    @classmethod
    @sentry_span
    async def _store_precomputed_branches(
        cls,
        account: int,
        branches: pd.DataFrame,
        repo_ids_to_update: Optional[Iterable[int]],
        pdb: DatabaseLike,
    ) -> None:
        order = np.argsort(branches[Branch.repository_node_id.name].values)
        repo_ids, counts = np.unique(
            branches[Branch.repository_node_id.name].values[order], return_counts=True,
        )
        if repo_ids_to_update is None:
            repo_indexes = np.arange(len(repo_ids), dtype=int)
        else:
            if not isinstance(repo_ids_to_update, (tuple, list, np.ndarray)):
                repo_ids_to_update = list(repo_ids_to_update)
            repo_indexes = np.flatnonzero(np.in1d(repo_ids, repo_ids_to_update))
        borders = np.zeros(len(counts) + 1, dtype=int)
        np.cumsum(counts, out=borders[1:])
        inserted = []
        format_version = GitHubBranches.__table__.columns[
            GitHubBranches.format_version.key
        ].default.arg
        now = datetime.now(timezone.utc)
        for repo_index in repo_indexes:
            indexes = order[borders[repo_index] : borders[repo_index + 1]]
            pb = PrecomputedBranches.from_fields(
                **{col: branches[col].values[indexes] for col in PrecomputedBranches.dtype.names},
            )
            inserted.append(
                GitHubBranches(
                    acc_id=account,
                    format_version=format_version,
                    repository_node_id=repo_ids[repo_index],
                    data=pb.data,
                    updated_at=now,
                ).explode(with_primary_keys=True),
            )
        sql = (await dialect_specific_insert(pdb))(GitHubBranches)
        sql = sql.on_conflict_do_update(
            index_elements=GitHubBranches.__table__.primary_key.columns,
            set_={
                col.name: getattr(sql.excluded, col.name)
                for col in (GitHubBranches.updated_at, GitHubBranches.data)
            },
        )
        with sentry_sdk.start_span(
            op=f"_store_precomputed_branches/execute_many({len(inserted)})",
        ):
            if pdb.url.dialect == "sqlite":
                async with pdb.connection() as pdb_conn:
                    async with pdb_conn.transaction():
                        await pdb_conn.execute_many(sql, inserted)
            else:
                # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
                await pdb.execute_many(sql, inserted)


def dummy_branches_df() -> pd.DataFrame:
    """Create an empty dataframe with Branch columns."""
    return pd.DataFrame(columns=[c.name for c in Branch.__table__.columns])
