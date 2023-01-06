from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Collection, Iterable, Iterator, Optional

import aiomcache
import numpy as np
import pandas as pd
from sqlalchemy import func, select

from athenian.api import metadata
from athenian.api.async_utils import read_sql_query
from athenian.api.cache import cached, cached_methods, middle_term_exptime
from athenian.api.db import DatabaseLike
from athenian.api.internal.logical_repos import coerce_logical_repos
from athenian.api.internal.prefixer import Prefixer
from athenian.api.models.metadata.github import Branch, NodeCommit, NodeRepositoryRef, Repository
from athenian.api.models.persistentdata.models import HealthMetric
from athenian.api.pandas_io import deserialize_args, serialize_args
from athenian.api.tracing import sentry_span


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
    @sentry_span
    @cached(
        exptime=middle_term_exptime,
        serialize=serialize_args,
        deserialize=deserialize_args,
        key=lambda meta_ids, repos, strip=False, **_: (
            ",".join(map(str, meta_ids)),
            ",".join(sorted(repos if repos is not None else ["None"])),
            strip,
        ),
    )
    async def load_branches(
        cls,
        repos: Optional[Iterable[str]],
        prefixer: Prefixer,
        meta_ids: tuple[int, ...],
        mdb: DatabaseLike,
        cache: Optional[aiomcache.Client],
        strip: bool = False,
        metrics: Optional[BranchMinerMetrics] = None,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        Fetch branches in the given repositories and extract the default branch names.

        :param strip: Value indicating whether the repository names are prefixed.
        :param since: Discard branches last updated before this date.
        :param metrics: Report error statistics by mutating this object.
        """
        if strip and repos is not None:
            repos = [r.split("/", 1)[1] for r in repos]
        if repos is not None:
            repos = coerce_logical_repos(repos)
            repo_ids = [prefixer.repo_name_to_node.get(r) for r in repos]
        else:
            repo_ids = None
        branches = await cls._fetch_branches(repo_ids, None, prefixer, meta_ids, mdb)
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
            commit_ids = np.concatenate([rb[0] for rb in ambiguous_defaults.values()])
            committed_dates = dict(
                await mdb.fetch_all(
                    select(NodeCommit.id, NodeCommit.committed_date).where(
                        NodeCommit.id.in_(commit_ids), NodeCommit.acc_id.in_(meta_ids),
                    ),
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
                    "resolved <ambiguous> default branch in %s to %s", repo, default_branch,
                )
                default_branches[repo] = default_branch
        zero_branch_repos = [repo for repo in repos if repo not in default_branches]
        if zero_branch_repos:
            existing_zero_branch_repos = dict(
                await mdb.fetch_all(
                    select(Repository.node_id, Repository.full_name).where(
                        Repository.full_name.in_(zero_branch_repos),
                        Repository.acc_id.in_(meta_ids),
                    ),
                ),
            )
            deleted_repos = set(zero_branch_repos) - set(existing_zero_branch_repos.values())
            if deleted_repos:
                for repo in deleted_repos:
                    default_branches[repo] = "master"
                log.error("some repositories do not exist: %s", deleted_repos)
            if existing_zero_branch_repos:
                rows = await mdb.fetch_all(
                    select(
                        NodeRepositoryRef.parent_id,
                        func.count(NodeRepositoryRef.child_id).label("numrefs"),
                    )
                    .where(
                        NodeRepositoryRef.acc_id.in_(meta_ids),
                        NodeRepositoryRef.parent_id.in_(existing_zero_branch_repos),
                    )
                    .group_by(NodeRepositoryRef.parent_id),
                )
                refs = {r[NodeRepositoryRef.parent_id.name]: r["numrefs"] for r in rows}
                reported_repos = set()
                warnings = []
                errors = []
                for node_id, full_name in existing_zero_branch_repos.items():
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
        return branches, default_branches

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
        query = (
            select(Branch)
            .where(
                *((Branch.repository_node_id.in_(repos),) if repos is not None else ()),
                Branch.acc_id.in_(meta_ids),
                *((Branch.commit_date >= since,) if since is not None else ()),
            )
            .with_statement_hint(
                "IndexOnlyScan(c node_commit_repository_target)"
                if since is None
                else "IndexOnlyScan(c github_node_commit_check_runs)",
            )
            .with_statement_hint("Rows(ref c repo *100)")
            .with_statement_hint("Rows(ref c repo rr *100)")
            .with_statement_hint(
                "Rows(ref c repo rr n1 n2 "
                f"*{len(repos) if repos is not None else len(prefixer.repo_node_to_name)})",
            )
        )
        if repos is not None:
            scan = (
                "IndexScan" if len(repos) < len(prefixer.repo_node_to_name) * 0.8 else "BitmapScan"
            )
            query = query.with_statement_hint(f"{scan}(ref node_ref_heads_repository_id)")
        if since is None:
            query = query.with_statement_hint("set(join_collapse_limit 1)")
        return await read_sql_query(query, mdb, Branch)


def dummy_branches_df() -> pd.DataFrame:
    """Create an empty dataframe with Branch columns."""
    return pd.DataFrame(columns=[c.name for c in Branch.__table__.columns])
