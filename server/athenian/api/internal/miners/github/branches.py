from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import compress
import logging
from typing import Collection, Iterable, Iterator, Optional

import aiomcache
import medvedi as md
from medvedi.accelerators import in1d_str
from medvedi.merge_to_str import merge_to_str
import numpy as np
import sentry_sdk
from sqlalchemy import delete, distinct, select, union_all

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import CancelCache, cached, cached_methods, short_term_exptime
from athenian.api.db import Database, DatabaseLike, dialect_specific_insert
from athenian.api.defer import defer
from athenian.api.internal.logical_repos import coerce_logical_repos
from athenian.api.internal.prefixer import Prefixer
from athenian.api.io import deserialize_args, serialize_args
from athenian.api.models.metadata.github import Branch, NodeRepositoryRef
from athenian.api.models.persistentdata.models import HealthMetric
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import numpy_struct
from athenian.precomputer.db.models import GitHubBranches, GitHubRepository


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
    missing: set[str]
    no_default: set[str]

    @classmethod
    def empty(cls) -> "BranchMinerMetrics":
        """Initialize a new BranchMinerMetrics instance filled with zeros."""
        return BranchMinerMetrics(0, set(), set())

    def as_db(self) -> Iterator[HealthMetric]:
        """Generate HealthMetric-s from this instance."""
        yield HealthMetric(name="branches_count", value=self.count)
        yield HealthMetric(name="branches_empty_count", value=len(self.missing))
        yield HealthMetric(name="branches_no_default", value=len(self.no_default))


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
        full_sync: bool = False,
        metrics: Optional[BranchMinerMetrics] = None,
    ) -> tuple[md.DataFrame, dict[str, str]]:
        """
        Fetch branches in the given repositories and extract the default branch names.

        :param strip: Value indicating whether the repository names are prefixed.
        :param full_sync: Value indicating whether we load branches from scratch and overwrite pdb.
        :param metrics: Report error statistics by mutating this object.
        """
        return (
            await cls._load_branches(
                repos, prefixer, account, meta_ids, mdb, pdb, cache, strip, full_sync, metrics,
            )
        )[:-1]

    def _postprocess_branches(
        result: tuple[md.DataFrame, dict[str, str], bool],
        repos: Optional[Iterable[str]],
        strip: bool,
        **_,
    ) -> tuple[md.DataFrame, dict[str, str], bool]:
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
        pdb: Optional[Database],
        cache: Optional[aiomcache.Client],
        strip: bool,
        full_sync: bool,
        metrics: Optional[BranchMinerMetrics],
    ) -> tuple[md.DataFrame, dict[str, str], bool]:
        if strip and repos is not None:
            repos = [r.split("/", 1)[1] for r in repos]
        if repos is not None:
            repos = coerce_logical_repos(repos)
            repo_ids = [prefixer.repo_name_to_node.get(r) for r in repos]
        else:
            repo_ids = None
        now = datetime.now(timezone.utc)
        if incremental := (pdb is not None and not full_sync):
            deadline = now - timedelta(hours=2)
        else:
            deadline = None
        new_branches, missed_branches, old_branches = await gather(
            cls._fetch_branches(repo_ids, deadline, prefixer, meta_ids, mdb),
            cls._fetch_branches_for_missing_repositories(
                repo_ids, deadline, account, prefixer, meta_ids, mdb, pdb,
            )
            if incremental
            else None,
            cls._fetch_precomputed_branches(account, repo_ids, prefixer, pdb)
            if incremental
            else None,
        )
        if missed_branches is not None and not missed_branches.empty:
            if new_branches.empty:
                new_branches = missed_branches
            else:
                new_mask = new_branches.isin(
                    Branch.branch_id.name,
                    missed_branches[Branch.branch_id.name],
                    invert=True,
                )
                if new_count := new_mask.sum():
                    if new_count < len(new_branches):
                        new_branches = new_branches.take(new_mask)
                    new_branches = md.concat(missed_branches, new_branches, ignore_index=True)
                else:
                    new_branches = missed_branches
        if old_branches is None or old_branches.empty:
            branches = new_branches
        elif new_branches.empty:
            branches = old_branches
        else:
            old_branch_names = old_branches[Branch.branch_name.name].astype("U", copy=False)
            new_branch_names = new_branches[Branch.branch_name.name].astype("U", copy=False)
            joint_index = merge_to_str(
                np.concatenate(
                    [
                        old_branches[Branch.repository_node_id.name],
                        new_branches[Branch.repository_node_id.name],
                    ],
                ),
                np.concatenate(
                    [
                        old_branch_names.view(f"S{old_branch_names.dtype.itemsize}"),
                        new_branch_names.view(f"S{new_branch_names.dtype.itemsize}"),
                    ],
                ),
            )
            old_mask = in1d_str(
                joint_index[: len(old_branches)],
                joint_index[len(old_branches) :],
                verbatim=True,
                invert=True,
            )
            if old_count := old_mask.sum():
                if old_count < len(old_branches):
                    old_branches = old_branches.take(old_mask)
                branches = md.concat(new_branches, old_branches, ignore_index=True)
            else:
                branches = new_branches
        if pdb is not None:
            await defer(
                cls._store_precomputed_branches(
                    account,
                    branches,
                    repo_ids,
                    new_branches.unique(Branch.repository_node_id.name, unordered=True)
                    if not new_branches.empty
                    else [],
                    now,
                    prefixer,
                    pdb,
                ),
                f"store_branches({len(branches)})",
            )
        log = logging.getLogger("%s.extract_default_branches" % metadata.__package__)
        default_branches = {}
        ambiguous_defaults = {}
        branch_repos = branches[Branch.repository_full_name.name]
        unique_branch_repos, index_map, counts = np.unique(
            branch_repos, return_inverse=True, return_counts=True,
        )
        if repos is None:
            repos = unique_branch_repos
        order = np.argsort(index_map)
        branch_names = branches[Branch.branch_name.name][order]
        branch_defaults = branches[Branch.is_default.name][order]
        branch_commit_ids = branches[Branch.commit_id.name][order]
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
                    metrics.no_default.add(repo)
            default_branches[repo] = default_branch
            pos = next_pos
        if ambiguous_defaults:
            map_indexes = np.flatnonzero(
                np.in1d(
                    branches[Branch.commit_id.name],
                    np.concatenate([rb[0] for rb in ambiguous_defaults.values()]),
                ),
            )
            committed_dates = dict(
                zip(
                    branches[Branch.commit_id.name][map_indexes],
                    branches[Branch.commit_date.name][map_indexes],
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
            if zero_names := zero_branch_repos & prefixer.repo_name_to_node.keys():
                for full_name in zero_names:
                    default_branches[full_name] = "master"
                zero_nodes = np.fromiter(
                    (prefixer.repo_name_to_node[k] for k in zero_names), int, len(zero_names),
                )
                refs = (
                    await read_sql_query(
                        select(distinct(NodeRepositoryRef.parent_id)).where(
                            NodeRepositoryRef.acc_id.in_(meta_ids),
                            NodeRepositoryRef.parent_id.in_(zero_nodes),
                        ),
                        mdb,
                        [NodeRepositoryRef.parent_id],
                    )
                )[NodeRepositoryRef.parent_id.name]
                inconsistent = np.in1d(zero_nodes, refs, assume_unique=True)
                if errors := list(compress(zero_names, inconsistent)):
                    if metrics is not None:
                        metrics.missing.update(errors)
                    log.error("the following repositories have 0 branches but >0 refs: %s", errors)
        if metrics is not None:
            metrics.count = len(branches)
        return branches, default_branches, repo_ids is not None

    _postprocess_branches = staticmethod(_postprocess_branches)

    @classmethod
    async def reset_caches(
        cls,
        account: int,
        pdb: Database,
        cache: Optional[aiomcache.Client],
    ) -> None:
        """Drop the cache of load_branches()."""
        await gather(
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
                False,
                False,
                None,
            ),
            pdb.execute(delete(GitHubBranches).where(GitHubBranches.acc_id == account)),
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
    ) -> md.DataFrame:
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
        if since is None or len(meta_ids) == 1:
            query = select(*columns).where(
                Branch.acc_id.in_(meta_ids),
                *((Branch.commit_date >= since,) if since is not None else ()),
                *((query_repo_col.in_(repos),) if repos is not None else ()),
            )
        else:
            query = union_all(
                *(
                    select(*columns).where(
                        Branch.acc_id == meta_id,
                        Branch.commit_date >= since,
                        *((query_repo_col.in_(repos),) if repos is not None else ()),
                    )
                    for meta_id in meta_ids
                ),
            )
        if since is None:
            if repos is not None:
                scan = (
                    "IndexScan"
                    if len(repos) < len(prefixer.repo_node_to_name) * 0.8
                    else "BitmapScan"
                )
                query = query.with_statement_hint(f"{scan}(ref node_ref_heads_repository_id)")
            # fmt: off
            query = (
                query
                .with_statement_hint("IndexOnlyScan(c node_commit_repository_target)")
                .with_statement_hint(
                    "Rows(ref c repo rr n1 n2 "
                    f"*{len(repos) if repos is not None else len(prefixer.repo_node_to_name)})",
                )
                .with_statement_hint("Rows(ref c repo *100)")
                .with_statement_hint("Rows(ref c repo rr *100)")
                .with_statement_hint("set(join_collapse_limit 1)")
            )
            # fmt: on
        else:
            query = query.with_statement_hint("Leading((((((c ref) rr) repo) n1) n2))")
            # Postgres may think we will fetch 0 rows, but that's OK on a narrow time window
            if repos is not None:
                # fmt: off
                query = (
                    query
                    .with_statement_hint(f"Rows(c ref *{len(repos) * 5})")
                    .with_statement_hint("IndexOnlyScan(c github_node_commit_check_runs)")
                )
                # fmt: on
            else:
                query = query.with_statement_hint("IndexScan(c github_node_commit_committed_date)")
        return await read_sql_query(query, mdb, columns)

    @classmethod
    async def _fetch_branches_for_missing_repositories(
        cls,
        repos: Collection[int],
        deadline: datetime,
        account: int,
        prefixer: Prefixer,
        meta_ids: tuple[int, ...],
        mdb: DatabaseLike,
        pdb: Database,
    ) -> md.DataFrame:
        branches_fetched_ats = dict(
            await pdb.fetch_all(
                select(GitHubRepository.node_id, GitHubRepository.branches_fetched_at).where(
                    GitHubRepository.acc_id == account,
                    *((GitHubRepository.node_id.in_(repos),) if repos is not None else ()),
                ),
            ),
        )
        new_deadline = deadline
        repos_to_update = []
        repos_to_fetch_from_scratch = []
        for repo in repos if repos is not None else prefixer.repo_node_to_name:
            try:
                new_deadline = min(new_deadline, branches_fetched_ats[repo])
            except (KeyError, TypeError):
                repos_to_fetch_from_scratch.append(repo)
            else:
                if new_deadline < deadline:
                    repos_to_update.append(repo)
        dfs = await gather(
            cls._fetch_branches(repos_to_fetch_from_scratch, None, prefixer, meta_ids, mdb)
            if repos_to_fetch_from_scratch
            else None,
            cls._fetch_branches(repos_to_update, new_deadline, prefixer, meta_ids, mdb)
            if repos_to_update
            else None,
        )
        if dfs[0] is None:
            if dfs[1] is None:
                return md.DataFrame()
            return dfs[1]
        if dfs[1] is None:
            return dfs[0]
        return md.concat(*dfs, ignore_index=True)

    @classmethod
    @sentry_span
    async def _fetch_precomputed_branches(
        cls,
        account: int,
        repos: Optional[Collection[int]],
        prefixer: Prefixer,
        pdb: Database,
    ) -> md.DataFrame:
        rows = await pdb.fetch_all(
            select(GitHubBranches.repository_node_id, GitHubBranches.data).where(
                GitHubBranches.acc_id == account,
                GitHubBranches.format_version
                == GitHubBranches.__table__.columns[GitHubBranches.format_version.key].default.arg,
                *((GitHubBranches.repository_node_id.in_(repos),) if repos is not None else ()),
            ),
        )
        if not rows:
            return md.DataFrame()
        repo_node_to_name = prefixer.repo_node_to_name.get
        branches = [
            PrecomputedBranches(r[1], repository_node_id=r[0], repository_full_name=name)
            for r in rows
            if (name := repo_node_to_name(r[0])) is not None
        ]
        columns = {}
        for col in PrecomputedBranches.dtype.names:
            try:
                values, borders = PrecomputedBranches.vectorize_field(branches, col)
            except NotImplementedError:
                values = np.concatenate([rb[col] for rb in branches])
            columns[col] = values
        lengths = np.diff(borders)
        columns[Branch.repository_node_id.name] = np.repeat(
            [b.repository_node_id for b in branches], lengths,
        )
        columns[Branch.repository_full_name.name] = np.repeat(
            [b.repository_full_name for b in branches], lengths,
        )
        return md.DataFrame(
            columns,
            dtype={
                Branch.repository_full_name.name: object,
                Branch.branch_name.name: object,
            },
        )

    @classmethod
    @sentry_span
    async def _store_precomputed_branches(
        cls,
        account: int,
        branches: md.DataFrame,
        all_repo_ids: Optional[list[int]],
        repo_ids_to_update: Collection[int],
        now: datetime,
        prefixer: Prefixer,
        pdb: Database,
    ) -> None:
        order = np.argsort(branches[Branch.repository_node_id.name])
        repo_ids, repo_id_ff, counts = np.unique(
            branches[Branch.repository_node_id.name][order],
            return_index=True,
            return_counts=True,
        )
        if len(repo_ids_to_update):
            if not isinstance(repo_ids_to_update, (tuple, list, np.ndarray)):
                repo_ids_to_update = list(repo_ids_to_update)
            repo_indexes = np.flatnonzero(np.in1d(repo_ids, repo_ids_to_update))
        else:
            repo_indexes = []
        borders = np.zeros(len(counts) + 1, dtype=int)
        np.cumsum(counts, out=borders[1:])
        inserted_branches = []
        inserted_repos = []
        format_version = GitHubBranches.__table__.columns[
            GitHubBranches.format_version.key
        ].default.arg
        for repo_index in repo_indexes:
            indexes = order[borders[repo_index] : borders[repo_index + 1]]
            pb = PrecomputedBranches.from_fields(
                **{col: branches[col][indexes] for col in PrecomputedBranches.dtype.names},
            )
            inserted_branches.append(
                GitHubBranches(
                    acc_id=account,
                    format_version=format_version,
                    repository_node_id=repo_ids[repo_index],
                    data=pb.data,
                    updated_at=now,
                ).explode(with_primary_keys=True),
            )
        if all_repo_ids is None:
            all_repo_ids = repo_ids
        repo_node_to_name = prefixer.repo_node_to_name.get
        for repo_id in all_repo_ids:
            inserted_repos.append(
                GitHubRepository(
                    acc_id=account,
                    node_id=repo_id,
                    repository_full_name=repo_node_to_name(repo_id),
                    branches_fetched_at=now,
                    updated_at=now,
                ).explode(with_primary_keys=True),
            )
        insert = await dialect_specific_insert(pdb)
        sql_branches = insert(GitHubBranches)
        sql_branches = sql_branches.on_conflict_do_update(
            index_elements=GitHubBranches.__table__.primary_key.columns,
            set_={
                col.name: getattr(sql_branches.excluded, col.name)
                for col in (GitHubBranches.updated_at, GitHubBranches.data)
            },
        )
        sql_repos = insert(GitHubRepository)
        sql_repos = sql_repos.on_conflict_do_update(
            index_elements=GitHubRepository.__table__.primary_key.columns,
            set_={
                col.name: getattr(sql_repos.excluded, col.name)
                for col in (GitHubRepository.updated_at, GitHubRepository.branches_fetched_at)
            },
        )
        with sentry_sdk.start_span(
            op=f"_store_precomputed_branches/execute_many({len(inserted_branches)})",
        ):
            async with pdb.connection() as pdb_conn:
                async with pdb_conn.transaction():
                    await pdb_conn.execute_many(sql_branches, inserted_branches)
                    await pdb_conn.execute_many(sql_repos, inserted_repos)


def dummy_branches_df() -> md.DataFrame:
    """Create an empty dataframe with Branch columns."""
    return md.DataFrame(columns=[c.name for c in Branch.__table__.columns])
