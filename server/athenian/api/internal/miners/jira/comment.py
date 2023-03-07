from collections import defaultdict
from typing import Iterable, Sequence

import sqlalchemy as sa
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.db import DatabaseLike, Row
from athenian.api.models.metadata.jira import Comment


async def fetch_issues_comments(
    issue_ids: Sequence[str],
    jira_acc_id: int,
    mdb: DatabaseLike,
    extra_columns: Iterable[InstrumentedAttribute] = (),
) -> dict[bytes, Row]:
    """Fetch the comments for the given issues, mapped by Issue id."""
    columns = set(extra_columns) | {Comment.issue_id}
    where = [
        Comment.acc_id == jira_acc_id,
        Comment.issue_id.progressive_in(issue_ids),
        Comment.is_deleted.is_(False),
    ]
    stmt = sa.select(*columns).where(*where)
    rows = await mdb.fetch_all(stmt)
    comments: dict = defaultdict(list)
    for r in rows:
        comments[r[Comment.issue_id.name].encode("utf-8")].append(r)
    return comments
