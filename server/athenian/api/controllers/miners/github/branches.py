from datetime import datetime, timezone
import logging
import pickle
from typing import Dict, Iterable, Optional, Tuple

import aiomcache
import numpy as np
import pandas as pd
from sqlalchemy import and_, select

from athenian.api import metadata
from athenian.api.async_utils import read_sql_query
from athenian.api.cache import cached, cached_methods
from athenian.api.models.metadata.github import Branch, NodeCommit, Repository
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import DatabaseLike


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
                               ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Fetch branches in the given repositories and extract the default branch names."""
        branches = await cls._extract_branches(repos, meta_ids, mdb)
        log = logging.getLogger("%s.extract_default_branches" % metadata.__package__)
        default_branches = {}
        ambiguous_defaults = {}
        for repo, repo_branches in branches.groupby(Branch.repository_full_name.key, sort=False):
            default_indexes = np.where(repo_branches[Branch.is_default.key].values)[0]
            branch_names = repo_branches[Branch.branch_name.key]  # type: pd.Series
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
            commit_ids = np.concatenate([rb[Branch.commit_id.key].values
                                         for rb in ambiguous_defaults.values()])
            committed_dates = await mdb.fetch_all(
                select([NodeCommit.id, NodeCommit.committed_date])
                .where(and_(NodeCommit.id.in_(commit_ids),
                            NodeCommit.acc_id.in_(meta_ids))))
            committed_dates = {r[0]: r[1] for r in committed_dates}
            for repo, repo_branches in ambiguous_defaults.items():
                default_branch = max_date = None
                for name, commit_id in zip(repo_branches[Branch.branch_name.key].values,
                                           repo_branches[Branch.commit_id.key].values):
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
                sql = """
                    SELECT parent_id, COUNT(child_id) AS numrefs
                    FROM github_node_repository_refs
                    WHERE parent_id IN (%s) AND acc_id %s
                    GROUP BY parent_id;
                """ % (", ".join("'%s'" % n for n in existing_zero_branch_repos),
                       ("= %d" % meta_ids[0])
                       if len(meta_ids) == 1
                       else ("IN (%s)" % ", ".join(str(i) for i in meta_ids)))
                rows = await mdb.fetch_all(sql)
                refs = {r["parent_id"]: r["numrefs"] for r in rows}
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
                                ) -> Tuple[pd.DataFrame, Dict[str, str]]:
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
    branch_commit_ids = branches[Branch.commit_id.key].values
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


# TODO: these have to be removed, these are here just for keeping backward-compatibility
# without the need to re-write already all the places these functions are called
extract_branches = BranchMiner.extract_branches
