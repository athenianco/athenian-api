"""Db related test utilities."""

from datetime import datetime, timezone
from functools import reduce
from typing import Any, cast, Union

import asyncpg
import sqlalchemy as sa
from sqlalchemy.engine.row import Row
from sqlalchemy.orm.decl_api import DeclarativeMeta
from sqlalchemy.sql.expression import ClauseElement, Insert, Selectable

from athenian.api.db import Database, DatabaseLike
from athenian.precomputer.db.base import BaseType, explode_model


def model_insert_stmt(model: BaseType, *, with_primary_keys=True) -> Insert:
    """Build the SQLAlchemy statement to insert the model."""
    table = cast(Selectable, type(model))

    values = explode_model(model, with_primary_keys=with_primary_keys)
    return sa.insert(table).values(values)


async def assert_missing_row(db: DatabaseLike, table: DeclarativeMeta, **kwargs: Any) -> None:
    """Assert that a row with the given properties doesn't exist."""
    where_clause = _build_table_where_clause(table, **kwargs)
    stmt = sa.select(table).where(where_clause)
    row = await db.fetch_one(stmt)
    assert row is None


async def assert_existing_row(
    db: DatabaseLike, table: DeclarativeMeta, **kwargs: Any,
) -> Union[asyncpg.Record, Row]:
    """Assert that a row with the given properties exists, and return the row."""
    where_clause = _build_table_where_clause(table, **kwargs)
    stmt = sa.select(table).where(where_clause)
    row = await db.fetch_one(stmt)
    assert row is not None
    return row


def db_datetime_equals(db: Database, db_dt: datetime, dt: datetime) -> bool:
    """Compare a datetime coming from db against another timezone aware datetime."""
    assert dt.tzinfo is not None
    if db.url.dialect == "sqlite":
        db_dt_with_tz = db_dt.replace(tzinfo=timezone.utc)
    else:
        db_dt_with_tz = db_dt
    return db_dt_with_tz == dt


def _build_table_where_clause(table: DeclarativeMeta, **kwargs: Any) -> ClauseElement:
    return reduce(
        lambda acc, item: sa.and_(acc, getattr(table, item[0]) == item[1]),
        kwargs.items(),
        cast(ClauseElement, sa.true()),
    )
