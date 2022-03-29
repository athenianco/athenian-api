from datetime import datetime, timezone
import logging
import pickle
from typing import Dict, Iterable, Optional, Tuple

import aiomcache
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import and_, func, select

from athenian.api import metadata
from athenian.api.async_utils import read_sql_query_with_join_collapse
from athenian.api.cache import cached, cached_methods
from athenian.api.controllers.logical_repos import coerce_logical_repos
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.db import DatabaseLike
from athenian.api.models.metadata.github import Branch, NodeCommit, NodeRepositoryRef, Repository
from athenian.api.tracing import sentry_span


@cached_methods
class BranchMiner:
    """Load information related to branches."""

    @classmethod
    @sentry_span
    @cached(
        exptime=60 * 60,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda repos, **_: (",".join(sorted(repos if repos is not None else [])),),
    )
    async def extract_branches(cls,
                               repos: Optional[Iterable[str]],
                               prefixer: Prefixer,
                               meta_ids: Tuple[int, ...],
                               mdb: DatabaseLike,
                               cache: Optional[aiomcache.Client],
                               strip: bool = False,
                               ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Fetch branches in the given repositories and extract the default branch names.

        :param strip: Value indicating whether the repository names are prefixed.
        """
        if strip and repos is not None:
            repos = [r.split("/", 1)[1] for r in repos]
        if repos is not None:
            repos = coerce_logical_repos(repos)
            repo_ids = [prefixer.repo_name_to_node.get(r) for r in repos]
        else:
            repo_ids = None
        branches = await cls._extract_branches(repo_ids, meta_ids, mdb)
        log = logging.getLogger("%s.extract_default_branches" % metadata.__package__)
        default_branches = {}
        ambiguous_defaults = {}
        branch_repos = branches[Branch.repository_full_name.name].values
        unique_branch_repos, index_map, counts = np.unique(
            branch_repos, return_inverse=True, return_counts=True)
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
                    repo, len(repo_branch_names), default_branch)
            default_branches[repo] = default_branch
            pos = next_pos
        if ambiguous_defaults:
            commit_ids = np.concatenate([rb[0] for rb in ambiguous_defaults.values()])
            committed_dates = dict(await mdb.fetch_all(
                select([NodeCommit.id, NodeCommit.committed_date])
                .where(and_(NodeCommit.id.in_(commit_ids),
                            NodeCommit.acc_id.in_(meta_ids)))))
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
                log.warning("resolved <ambiguous> default branch in %s to %s",
                            repo, default_branch)
                default_branches[repo] = default_branch
        zero_branch_repos = [repo for repo in repos if repo not in default_branches]
        if zero_branch_repos:
            existing_zero_branch_repos = dict(
                await mdb.fetch_all(select([Repository.node_id, Repository.full_name])
                                    .where(and_(Repository.full_name.in_(zero_branch_repos),
                                                Repository.acc_id.in_(meta_ids)))))
            deleted_repos = set(zero_branch_repos) - set(existing_zero_branch_repos.values())
            if deleted_repos:
                for repo in deleted_repos:
                    default_branches[repo] = "master"
                log.error("some repositories do not exist: %s", deleted_repos)
            if existing_zero_branch_repos:
                rows = await mdb.fetch_all(
                    select([NodeRepositoryRef.parent_id,
                            func.count(NodeRepositoryRef.child_id).label("numrefs")])
                    .where(and_(NodeRepositoryRef.acc_id.in_(meta_ids),
                                NodeRepositoryRef.parent_id.in_(existing_zero_branch_repos)))
                    .group_by(NodeRepositoryRef.parent_id))
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
        return branches, default_branches

    @classmethod
    @sentry_span
    async def _extract_branches(cls,
                                repos: Optional[Iterable[int]],
                                meta_ids: Tuple[int, ...],
                                mdb: DatabaseLike,
                                ) -> pd.DataFrame:
        query = select([Branch]).where(and_(
            Branch.repository_node_id.in_(repos)
            if repos is not None
            else sa.true(),
            Branch.acc_id.in_(meta_ids))) \
            .with_statement_hint("IndexOnlyScan(c node_commit_repository_target)")
        df = await read_sql_query_with_join_collapse(query, mdb, Branch)
        for left_join_col in (Branch.commit_sha.name, Branch.repository_full_name.name):
            if (not_null := df[left_join_col].notnull().values).sum() < len(df):
                df = df.take(np.flatnonzero(not_null))
        return df


async def load_branch_commit_dates(branches: pd.DataFrame,
                                   meta_ids: Tuple[int, ...],
                                   mdb: DatabaseLike,
                                   ) -> None:
    """Fetch the branch head commit dates if needed. The operation executes in-place."""
    if Branch.commit_date in branches:
        return
    if branches.empty:
        branches[Branch.commit_date] = []
        return
    branch_commit_ids = branches[Branch.commit_id.name].values
    branch_commit_dates = dict(await mdb.fetch_all(
        select([NodeCommit.id, NodeCommit.committed_date])
        .where(and_(NodeCommit.id.in_(branch_commit_ids),
                    NodeCommit.acc_id.in_(meta_ids)))))
    if mdb.url.dialect == "sqlite":
        branch_commit_dates = {k: v.replace(tzinfo=timezone.utc)
                               for k, v in branch_commit_dates.items()}
    now = datetime.now(timezone.utc)
    branches[Branch.commit_date] = [branch_commit_dates.get(commit_id, now)
                                    for commit_id in branch_commit_ids]


def dummy_branches_df() -> pd.DataFrame:
    """Create an empty dataframe with Branch columns."""
    return pd.DataFrame(columns=[c.name for c in Branch.__table__.columns])
