from datetime import datetime, timezone
import logging
import pickle
from typing import Dict, Iterable, Optional, Tuple

import aiomcache
import numpy as np
import pandas as pd
from sqlalchemy import and_, func, select

from athenian.api import metadata
from athenian.api.async_utils import read_sql_query
from athenian.api.cache import cached, cached_methods
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
        key=lambda repos, **_: (",".join(sorted(repos)),),
    )
    async def extract_branches(cls,
                               repos: Iterable[str],
                               meta_ids: Tuple[int, ...],
                               mdb: DatabaseLike,
                               cache: Optional[aiomcache.Client],
                               strip: bool = False,
                               ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Fetch branches in the given repositories and extract the default branch names.

        :param strip: Value indicating whether the repository names are prefixed.
        """
        if strip:
            repos = [r.split("/", 1)[1] for r in repos]
        branches = await cls._extract_branches(repos, meta_ids, mdb)
        log = logging.getLogger("%s.extract_default_branches" % metadata.__package__)
        default_branches = {}
        ambiguous_defaults = {}
        for repo, repo_branches in branches.groupby(Branch.repository_full_name.name, sort=False):
            default_indexes = np.where(repo_branches[Branch.is_default.name].values)[0]
            branch_names = repo_branches[Branch.branch_name.name]  # type: pd.Series
            if len(default_indexes) > 0:
                default_branch = branch_names._ixs(default_indexes[0])
            else:
                branch_names = branch_names.values
                if "master" in branch_names:
                    default_branch = "master"
                elif "main" in branch_names:
                    default_branch = "main"
                elif len(branch_names) == 1:
                    default_branch = branch_names[0]
                elif len(branch_names) > 0:
                    ambiguous_defaults[repo] = repo_branches
                    default_branch = "<ambiguous>"
                else:
                    default_branch = "master"
                log.warning(
                    "%s does not have an explicit default branch among %d listed, set to %s",
                    repo, len(branch_names), default_branch)
            default_branches[repo] = default_branch
        if ambiguous_defaults:
            commit_ids = np.concatenate([rb[Branch.commit_id.name].values
                                         for rb in ambiguous_defaults.values()])
            committed_dates = await mdb.fetch_all(
                select([NodeCommit.id, NodeCommit.committed_date])
                .where(and_(NodeCommit.id.in_(commit_ids),
                            NodeCommit.acc_id.in_(meta_ids))))
            committed_dates = {r[0]: r[1] for r in committed_dates}
            for repo, repo_branches in ambiguous_defaults.items():
                default_branch = max_date = None
                for name, commit_id in zip(repo_branches[Branch.branch_name.name].values,
                                           repo_branches[Branch.commit_id.name].values):
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
            rows = await mdb.fetch_all(select([Repository.node_id, Repository.full_name])
                                       .where(and_(Repository.full_name.in_(zero_branch_repos),
                                                   Repository.acc_id.in_(meta_ids))))
            existing_zero_branch_repos = {r[0]: r[1] for r in rows}
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
                for node_id, full_name in existing_zero_branch_repos.items():
                    if full_name not in reported_repos:
                        (log.warning if refs.get(node_id, 0) == 0 else log.error)(
                            "repository %s has 0 branches", full_name)
                        default_branches[full_name] = "master"
                        reported_repos.add(full_name)
        return branches, default_branches

    @classmethod
    async def _extract_branches(cls,
                                repos: Iterable[str],
                                meta_ids: Tuple[int, ...],
                                mdb: DatabaseLike,
                                ) -> pd.DataFrame:
        return await read_sql_query(
            select([Branch]).where(and_(Branch.repository_full_name.in_(repos),
                                        Branch.acc_id.in_(meta_ids),
                                        Branch.commit_sha.isnot(None))),
            mdb, Branch)


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
    rows = await mdb.fetch_all(
        select([NodeCommit.id, NodeCommit.committed_date])
        .where(and_(NodeCommit.id.in_(branch_commit_ids),
                    NodeCommit.acc_id.in_(meta_ids))))
    branch_commit_dates = {r[0]: r[1] for r in rows}
    if mdb.url.dialect == "sqlite":
        branch_commit_dates = {k: v.replace(tzinfo=timezone.utc)
                               for k, v in branch_commit_dates.items()}
    now = datetime.now(timezone.utc)
    branches[Branch.commit_date] = [branch_commit_dates.get(commit_id, now)
                                    for commit_id in branch_commit_ids]


def dummy_branches_df() -> pd.DataFrame:
    """Create an empty dataframe with Branch columns."""
    return pd.DataFrame(columns=[c.name for c in Branch.__table__.columns])
