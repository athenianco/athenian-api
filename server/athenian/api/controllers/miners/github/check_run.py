from datetime import datetime
import pickle
from typing import Collection, Optional, Tuple

import aiomcache
import pandas as pd
from sqlalchemy import and_, select

from athenian.api.async_utils import read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import JIRAFilter
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.models.metadata.github import CheckRun
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import DatabaseLike


@sentry_span
@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repositories, commit_authors, jira, **_:  # noqa
    (
        time_from.timestamp(), time_to.timestamp(),
        ",".join(sorted(repositories)),
        ",".join(sorted(commit_authors)),
        jira,
    ),
)
async def mine_check_runs(time_from: datetime,
                          time_to: datetime,
                          repositories: Collection[str],
                          commit_authors: Collection[str],
                          jira: JIRAFilter,
                          meta_ids: Tuple[int, ...],
                          mdb: DatabaseLike,
                          cache: Optional[aiomcache.Client],
                          ) -> pd.DataFrame:
    """
    Filter check runs according to the specified parameters.

    :param time_from: Check runs must start later than this time.
    :param time_to: Check runs must start earlier than this time.
    :param repositories: Look for check runs in these repository names.
    :param commit_authors: Check runs must link to the commits with the given author logins.
    :param jira: Check runs must link to PRs satisfying this JIRA filter.
    :return: Pandas DataFrame with columns mapped from CheckRun model.
    """
    filters = [
        CheckRun.acc_id.in_(meta_ids),
        CheckRun.started_at.between(time_from, time_to),
        CheckRun.repository_full_name.in_(repositories),
    ]
    if commit_authors:
        filters.append(CheckRun.author_login.in_(commit_authors))
    if not jira:
        query = select([CheckRun]).where(and_(*filters))
    else:
        query = await generate_jira_prs_query(
            filters, jira, mdb, cache, columns=CheckRun.__table__.columns, seed=CheckRun,
            on=(CheckRun.pull_request_node_id, CheckRun.acc_id))
    return await read_sql_query(query, mdb, columns=CheckRun)
