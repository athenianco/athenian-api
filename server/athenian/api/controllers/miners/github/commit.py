from datetime import datetime
from enum import Enum
import pickle
from typing import Collection, List, Optional, Union

import aiomcache
import databases
from sqlalchemy import and_, outerjoin, select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import cached
from athenian.api.models.metadata.github import NodePullRequestCommit, PushCommit


class FilterCommitsProperty(Enum):
    """Primary commit filter modes."""

    NO_PR_MERGES = "no_pr_merges"
    BYPASSING_PRS = "bypassing_prs"


@cached(
    exptime=5 * 60,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda prop, date_from, date_to, repos, with_author, with_committer, columns, **_:
        (prop.value, date_from.timestamp(), date_to.timestamp(), ",".join(sorted(repos)),
         ",".join(sorted(with_author)) if with_author else "",
         ",".join(sorted(with_committer)) if with_committer else "",
         "" if columns is None else ",".join(c.key for c in columns)),
)
async def extract_commits(prop: FilterCommitsProperty,
                          date_from: datetime,
                          date_to: datetime,
                          repos: Collection[str],
                          with_author: Optional[Collection[str]],
                          with_committer: Optional[Collection[str]],
                          db: Union[databases.Database, databases.core.Connection],
                          cache: Optional[aiomcache.Client],
                          columns: Optional[List[InstrumentedAttribute]] = None,
                          ):
    """Fetch commits that satisfy the given filters."""
    assert isinstance(date_from, datetime)
    assert isinstance(date_to, datetime)
    sql_filters = [
        PushCommit.committed_date.between(date_from, date_to),
        PushCommit.repository_full_name.in_(repos),
        PushCommit.committer_email != "noreply@github.com",
    ]
    if with_author:
        sql_filters.append(PushCommit.author_login.in_(with_author))
    if with_committer:
        sql_filters.append(PushCommit.committer_login.in_(with_committer))
    if columns is None:
        cols_query, cols_df = [PushCommit], PushCommit
    else:
        cols_query = cols_df = columns
    if prop == FilterCommitsProperty.NO_PR_MERGES:
        commits = await read_sql_query(select(cols_query).where(and_(*sql_filters)), db, cols_df)
    elif prop == FilterCommitsProperty.BYPASSING_PRS:
        commits = await read_sql_query(
            select(cols_query)
            .select_from(outerjoin(PushCommit, NodePullRequestCommit,
                                   PushCommit.node_id == NodePullRequestCommit.commit))
            .where(and_(NodePullRequestCommit.commit.is_(None), *sql_filters)),
            db, cols_df)
    else:
        raise AssertionError('Unsupported primary commit filter "%s"' % prop)
    return commits
