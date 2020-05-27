import logging
import pickle
from typing import Dict, Iterable, Optional, Tuple

import aiomcache
import pandas as pd
from sqlalchemy import select

from athenian.api import metadata
from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import cached
from athenian.api.models.metadata.github import Branch
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
        select([Branch]).where(Branch.repository_full_name.in_(repos)), db, Branch)
    log = logging.getLogger("%s.extract_default_branches" % metadata.__package__)
    default_branches = {}
    for repo, repo_branches in branches.groupby(Branch.repository_full_name.key):
        try:
            default_branch = \
                repo_branches[Branch.branch_name.key][repo_branches[Branch.is_default.key]].iloc[0]
        except (IndexError, ValueError):
            log.error('failed to find the default branch for "%s": only have %s',
                      repo, repo_branches[[Branch.branch_name.key, Branch.is_default.key]])
            continue
        default_branches[repo] = default_branch
    return branches, default_branches
