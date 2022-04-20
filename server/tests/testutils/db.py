"""Db related test utilities."""

from functools import reduce
from typing import Any, cast

import sqlalchemy as sa
from sqlalchemy.orm.decl_api import DeclarativeMeta
from sqlalchemy.sql.expression import ClauseElement, Insert, Selectable

from athenian.api.db import DatabaseLike
from athenian.precomputer.db.base import BaseType


def model_insert_stmt(model: BaseType, *, with_primary_keys=True) -> Insert:
    """Build the SQLAlchemy statement to insert the model."""
    table = cast(Selectable, type(model))
    return sa.insert(table).values(model.explode(with_primary_keys=with_primary_keys))


async def assert_missing_row(db: DatabaseLike, table: DeclarativeMeta, **kwargs: Any) -> None:
    """Assert that a row with the given properties doesn't exist."""
    where_clause = reduce(
        lambda acc, item: sa.and_(acc, getattr(table, item[0]) == item[1]),
        kwargs.items(),
        cast(ClauseElement, sa.true()),
    )

    stmt = sa.select(table).where(where_clause)

    row = await db.fetch_one(stmt)
    assert row is None
