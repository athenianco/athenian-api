from typing import Iterable, Sequence

import medvedi as md
import sqlalchemy as sa
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.async_utils import read_sql_query
from athenian.api.db import DatabaseLike
from athenian.api.models.metadata.jira import Comment


async def fetch_issues_comments(
    issue_ids: Sequence[str],
    jira_acc_id: int,
    mdb: DatabaseLike,
    extra_columns: Iterable[InstrumentedAttribute] = (),
) -> md.DataFrame:
    """Fetch the comments for the given issues."""
    columns = set(extra_columns) | {Comment.issue_id}
    where = [
        Comment.acc_id == jira_acc_id,
        Comment.issue_id.progressive_in(issue_ids),
        Comment.is_deleted.is_(False),
    ]
    stmt = sa.select(*columns).where(*where)
    return await read_sql_query(stmt, mdb, list(columns))
