"""DB related test utilities."""

from __future__ import annotations

import contextlib
from typing import Any, AsyncIterator, Optional, Sequence, cast

import sqlalchemy as sa
from sqlalchemy.engine.row import Row
from sqlalchemy.orm.decl_api import DeclarativeMeta
from sqlalchemy.schema import Table
from sqlalchemy.sql.expression import ClauseElement, Insert, Selectable

from athenian.api.db import Connection, Database, DatabaseLike
from athenian.precomputer.db.base import BaseType, explode_model


def model_insert_stmt(model: BaseType, *, with_primary_keys=True) -> Insert:
    """Build the SQLAlchemy statement to insert the model."""
    table = cast(Selectable, type(model))

    values = explode_model(model, with_primary_keys=with_primary_keys)
    values = {k: v for k, v in values.items() if v is not SKIP_MODEL_FIELD}
    return sa.insert(table).values(values)


async def models_insert(db: Database, *models: BaseType) -> None:
    """Insert a set of models into a DB."""
    async with db.connection() as conn:
        for model in models:
            await conn.execute(model_insert_stmt(model))


async def models_insert_auto_pk(db: Database, *models: BaseType) -> list[Any]:
    """Insert a new model, letting DB handle primary key generation. Inserted PK is returned."""

    results = []
    async with db.connection() as conn:
        for model in models:
            stmt = model_insert_stmt(model, with_primary_keys=False)
            results.append(await conn.execute(stmt))
    return results


SKIP_MODEL_FIELD = object()
"""Set a model field to this value avoid propagating it to DB on model insertion.

For instance, this is needed to insert a SQL NULL into a JSONB column.
"""


async def assert_missing_row(db: DatabaseLike, table: DeclarativeMeta, **kwargs: Any) -> None:
    """Assert that a row with the given properties doesn't exist."""
    stmt = sa.select(table)
    if kwargs:
        stmt = stmt.where(_build_table_where_clause(table, **kwargs))
    row = await db.fetch_one(stmt)
    assert row is None


async def assert_existing_row(db: DatabaseLike, table: DeclarativeMeta, **kwargs: Any) -> Row:
    """Assert that a single row with the given properties exists, and return the row."""
    rows = await assert_existing_rows(db, table, **kwargs)
    if len(rows) > 1:
        raise AssertionError("More than one row returned")
    return rows[0]


async def assert_existing_rows(
    db: DatabaseLike,
    table: DeclarativeMeta,
    **kwargs: Any,
) -> Sequence[Row]:
    stmt = sa.select(table)
    if kwargs:
        stmt = stmt.where(_build_table_where_clause(table, **kwargs))
    rows = await db.fetch_all(stmt)
    if not rows:
        raise AssertionError("No row returned")
    return rows


async def count(db: Database, table: Table, where: Optional[ClauseElement] = None) -> int:
    """Execute a simple count query."""
    stmt = sa.select(sa.func.count()).select_from(table)
    if where is not None:
        stmt = stmt.where(where)
    return await db.fetch_val(stmt)


class DBCleaner:
    """Helper to delete objects from db at the end of tests.

    Objects to delete can be added with add_* methods, they will be deleted when
    the DBCleaner context manager exists.
    """

    def __init__(self, db: Database) -> None:
        assert isinstance(db, Database)
        assert getattr(db, "is_rw", False), "must use the mdb_rw fixture"
        self._db = db
        self._to_clean: list[tuple[DeclarativeMeta, dict]] = []

    def add_condition(self, model_cls: DeclarativeMeta, **params: Any) -> None:
        """Add a condition for a deletion on a table."""
        self._to_clean.append((model_cls, params))

    def add_model(self, model: Any) -> None:
        """Add a model to delete."""
        params = {
            k: getattr(model, k) for k, v in model.__table__.columns.items() if v.primary_key
        }
        self.add_condition(model.__class__, **params)

    def add_models(self, *models: Any) -> None:
        """Add multiple models to delete."""
        for model in models:
            self.add_model(model)

    async def __aenter__(self) -> DBCleaner:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> Optional[bool]:
        await self._clean()
        return None

    async def _clean(self) -> None:
        for table, params in self._to_clean:
            stmt = sa.delete(table)
            if params:
                stmt = stmt.where(_build_table_where_clause(table, **params))
            await self._db.execute(stmt)


@contextlib.asynccontextmanager
async def transaction_conn(db: Database) -> AsyncIterator[Connection]:
    """Async context manager to have a connection with a transaction started."""
    async with db.connection() as conn:
        async with conn.transaction():
            yield conn


def _build_table_where_clause(table: DeclarativeMeta, **kwargs: Any) -> ClauseElement:
    assert kwargs
    where = sa.and_(*(getattr(table, key) == val for key, val in kwargs.items()))
    return cast(ClauseElement, where)
