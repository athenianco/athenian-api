import logging
import pickle
from typing import Dict, Iterable, Optional, Tuple

import aiomcache
import pandas as pd
from sqlalchemy import and_, select

from athenian.api import metadata
from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import cached
from athenian.api.models.metadata.github import Branch, Repository
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import DatabaseLike


@sentry_span
@cached(
    exptime=60 * 60,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, **_: (",".join(sorted(repos)),),
)
async def extract_branches(repos: Iterable[str],
                           db: DatabaseLike,
                           cache: Optional[aiomcache.Client],
                           ) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Fetch branches in the given repositories and extract the default branch names."""
    branches = await read_sql_query(
        select([Branch]).where(and_(Branch.repository_full_name.in_(repos),
                                    Branch.commit_sha.isnot(None))), db, Branch)
    log = logging.getLogger("%s.extract_default_branches" % metadata.__package__)
    default_branches = {}
    for repo, repo_branches in branches.groupby(Branch.repository_full_name.key):
        try:
            default_branch = \
                repo_branches[Branch.branch_name.key][repo_branches[Branch.is_default.key]].iloc[0]
        except (IndexError, ValueError):
            log.error('failed to find the default branch for "%s": only have %s',
                      repo, repo_branches[[Branch.branch_name.key, Branch.is_default.key]])
            default_branch = "master"
        default_branches[repo] = default_branch
    zero_branch_repos = [repo for repo in repos if repo not in default_branches]
    if zero_branch_repos:
        rows = await db.fetch_all(select([Repository.node_id, Repository.full_name])
                                  .where(Repository.full_name.in_(zero_branch_repos)))
        existing_zero_branch_repos = {r[0]: r[1] for r in rows}
        deleted_repos = set(zero_branch_repos) - set(existing_zero_branch_repos)
        if deleted_repos:
            for repo in deleted_repos:
                default_branches[repo] = "master"
            log.error("some repositories do not exist: %s", deleted_repos)
        if existing_zero_branch_repos:
            sql = """
                SELECT parent_id, COUNT(child_id) AS numrefs
                FROM github_node_repository_refs
                WHERE parent_id IN (%s)
                GROUP BY parent_id;
            """ % ", ".join("'%s'" % n for n in existing_zero_branch_repos)
            rows = await db.fetch_all(sql)
            refs = {r["parent_id"]: r["numrefs"] for r in rows}
            for node_id, full_name in existing_zero_branch_repos.items():
                (log.warning if refs.get(node_id, 0) == 0 else log.error)(
                    "repository %s has 0 branches", full_name)
                default_branches[full_name] = "master"
    return branches, default_branches
