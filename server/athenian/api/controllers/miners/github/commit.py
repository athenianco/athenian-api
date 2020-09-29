from datetime import datetime
from enum import Enum
import logging
import pickle
from typing import Collection, List, Optional

import aiomemcached
import numpy as np
import sentry_sdk
from sqlalchemy import and_, outerjoin, select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import cached
from athenian.api.models.metadata.github import NodePullRequestCommit, PushCommit, User
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import DatabaseLike


class FilterCommitsProperty(Enum):
    """Primary commit filter modes."""

    NO_PR_MERGES = "no_pr_merges"
    BYPASSING_PRS = "bypassing_prs"


@sentry_span
@cached(
    exptime=5 * 60,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda prop, date_from, date_to, repos, with_author, with_committer, **kwargs:
        (prop.value, date_from.timestamp(), date_to.timestamp(), ",".join(sorted(repos)),
         ",".join(sorted(with_author)) if with_author else "",
         ",".join(sorted(with_committer)) if with_committer else "",
         "" if kwargs.get("columns") is None else ",".join(c.key for c in kwargs["columns"])),
)
async def extract_commits(prop: FilterCommitsProperty,
                          date_from: datetime,
                          date_to: datetime,
                          repos: Collection[str],
                          with_author: Optional[Collection[str]],
                          with_committer: Optional[Collection[str]],
                          db: DatabaseLike,
                          cache: Optional[aiomemcached.Client],
                          columns: Optional[List[InstrumentedAttribute]] = None,
                          ):
    """Fetch commits that satisfy the given filters."""
    assert isinstance(date_from, datetime)
    assert isinstance(date_to, datetime)
    log = logging.getLogger("%s.extract_commits" % metadata.__package__)
    sql_filters = [
        PushCommit.committed_date.between(date_from, date_to),
        PushCommit.repository_full_name.in_(repos),
        PushCommit.committer_email != "noreply@github.com",
    ]
    user_logins = set()
    if with_author:
        user_logins.update(with_author)
    if with_committer:
        user_logins.update(with_committer)
    if user_logins:
        rows = await db.fetch_all(
            select([User.login, User.node_id]).where(User.login.in_(user_logins)))
        user_ids = {r[0]: r[1] for r in rows}
        del user_logins
    else:
        user_ids = {}
    if with_author:
        author_ids = []
        for u in with_author:
            try:
                author_ids.append(user_ids[u])
            except KeyError:
                continue
        sql_filters.append(PushCommit.author_user.in_(author_ids))
    if with_committer:
        committer_ids = []
        for u in with_committer:
            try:
                committer_ids.append(user_ids[u])
            except KeyError:
                continue
        sql_filters.append(PushCommit.committer_user.in_(committer_ids))
    if columns is None:
        cols_query, cols_df = [PushCommit], PushCommit
    else:
        if PushCommit.node_id not in columns:
            columns.append(PushCommit.node_id)
        cols_query = cols_df = columns
    if prop == FilterCommitsProperty.NO_PR_MERGES:
        with sentry_sdk.start_span(op="extract_commits/fetch/NO_PR_MERGES"):
            commits = await read_sql_query(
                select(cols_query).where(and_(*sql_filters)), db, cols_df)
    elif prop == FilterCommitsProperty.BYPASSING_PRS:
        with sentry_sdk.start_span(op="extract_commits/fetch/BYPASSING_PRS"):
            commits = await read_sql_query(
                select(cols_query)
                .select_from(outerjoin(PushCommit, NodePullRequestCommit,
                                       PushCommit.node_id == NodePullRequestCommit.commit))
                .where(and_(NodePullRequestCommit.commit.is_(None), *sql_filters)),
                db, cols_df)
    else:
        raise AssertionError('Unsupported primary commit filter "%s"' % prop)
    for number_prop in (PushCommit.additions, PushCommit.deletions, PushCommit.changed_files):
        try:
            number_col = commits[number_prop.key]
        except KeyError:
            continue
        nans = commits[PushCommit.node_id.key].take(np.where(number_col.isna())[0])
        if not nans.empty:
            log.error("[DEV-546] Commits have NULL in %s: %s", number_prop.key, ", ".join(nans))
    return commits
