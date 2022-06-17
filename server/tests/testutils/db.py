"""Db related test utilities."""

from functools import reduce
from typing import Any, Optional, Union, cast

import asyncpg
import sqlalchemy as sa
from sqlalchemy.engine.row import Row
from sqlalchemy.orm.decl_api import DeclarativeMeta
from sqlalchemy.schema import Table
from sqlalchemy.sql.expression import ClauseElement, Insert, Selectable

from athenian.api.db import Database, DatabaseLike
from athenian.precomputer.db.base import BaseType, explode_model


def model_insert_stmt(model: BaseType, *, with_primary_keys=True) -> Insert:
    """Build the SQLAlchemy statement to insert the model."""
    table = cast(Selectable, type(model))

    values = explode_model(model, with_primary_keys=with_primary_keys)
    return sa.insert(table).values(values)


async def models_insert(db: Database, *models: BaseType) -> None:
    """Insert a set of models into a DB."""
    async with db.connection() as conn:
        for model in models:
            await conn.execute(model_insert_stmt(model))


async def assert_missing_row(db: DatabaseLike, table: DeclarativeMeta, **kwargs: Any) -> None:
    """Assert that a row with the given properties doesn't exist."""
    where_clause = _build_table_where_clause(table, **kwargs)
    stmt = sa.select(table).where(where_clause)
    row = await db.fetch_one(stmt)
    assert row is None


async def assert_existing_row(
    db: DatabaseLike,
    table: DeclarativeMeta,
    **kwargs: Any,
) -> Union[asyncpg.Record, Row]:
    """Assert that a row with the given properties exists, and return the row."""
    where_clause = _build_table_where_clause(table, **kwargs)
    stmt = sa.select(table).where(where_clause)
    row = await db.fetch_one(stmt)
    assert row is not None
    return row


async def count(db: Database, table: Table, where: Optional[ClauseElement] = None) -> int:
    """Execute a simple count query."""
    stmt = sa.select(sa.func.count()).select_from(table)
    if where is not None:
        stmt = stmt.where(where)
    return await db.fetch_val(stmt)


def _build_table_where_clause(table: DeclarativeMeta, **kwargs: Any) -> ClauseElement:
    return reduce(
        lambda acc, item: sa.and_(acc, getattr(table, item[0]) == item[1]),
        kwargs.items(),
        cast(ClauseElement, sa.true()),
    )
