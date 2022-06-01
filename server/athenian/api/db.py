import asyncio
from contextvars import ContextVar
import logging
import pickle
import re
import sqlite3
import threading
import time
from typing import Any, Callable, List, Mapping, Optional, Union
from urllib.parse import quote

import aiohttp.web
import aiosqlite
import asyncpg
from asyncpg.rkt import set_query_dtype
from flogging import flogging
import morcilla.core
from morcilla.interfaces import ConnectionBackend, TransactionBackend
import numpy as np
import sentry_sdk
from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import CompoundSelect, Select
from sqlalchemy.sql.functions import ReturnTypeFromArgs

import athenian
from athenian.api import metadata
from athenian.api.models import check_alembic_schema_version, check_collation, \
    DBSchemaMismatchError
from athenian.api.models.metadata import check_schema_version as check_mdb_schema_version
from athenian.api.slogging import log_multipart
from athenian.api.typing_utils import wraps


def add_pdb_metrics_context(app: aiohttp.web.Application) -> dict:
    """Create and attach the precomputed DB metrics context."""
    ctx = app["pdb_context"] = {
        "hits": ContextVar("pdb_hits", default=None),
        "misses": ContextVar("pdb_misses", default=None),
    }
    return ctx


def _repr_instrumented_attribute(attr: InstrumentedAttribute) -> str:
    return f"<{attr.property}>"


InstrumentedAttribute.__sentry_repr__ = _repr_instrumented_attribute

flogging.trailing_dot_exceptions.add("asyncpg.pool")
pdb_metrics_logger = logging.getLogger("%s.pdb" % metadata.__package__)


def set_pdb_hits(pdb: morcilla.Database, topic: str, value: int) -> None:
    """Assign the `topic` precomputed DB hits to `value`."""
    pdb.metrics["hits"].get()[topic] = value
    pdb_metrics_logger.info("hits/%s: %d", topic, value, stacklevel=2)


def set_pdb_misses(pdb: morcilla.Database, topic: str, value: int) -> None:
    """Assign the `topic` precomputed DB misses to `value`."""
    pdb.metrics["misses"].get()[topic] = value
    pdb_metrics_logger.info("misses/%s: %d", topic, value, stacklevel=2)


def add_pdb_hits(pdb: morcilla.Database, topic: str, value: int) -> None:
    """Increase the `topic` precomputed hits by `value`."""
    if value < 0:
        pdb_metrics_logger.error('negative add_pdb_hits("%s", %d)', topic, value)
    pdb.metrics["hits"].get()[topic] += value
    pdb_metrics_logger.info("hits/%s: +%d", topic, value, stacklevel=2)


def add_pdb_misses(pdb: morcilla.Database, topic: str, value: int) -> None:
    """Increase the `topic` precomputed misses by `value`."""
    if value < 0:
        pdb_metrics_logger.error('negative add_pdb_misses("%s", %d)', topic, value)
    pdb.metrics["misses"].get()[topic] += value
    pdb_metrics_logger.info("misses/%s: +%d", topic, value, stacklevel=2)


Connection = morcilla.Connection
Database = morcilla.Database
Row = Mapping[Union[int, str], Any]


_sql_log = logging.getLogger("%s.sql" % metadata.__package__)
_sql_str_re = re.compile(r"'[^']+'(, )?")
_log_sql_re = re.compile(r"SELECT|\(SELECT|WITH RECURSIVE")


def _generate_tags() -> str:
    with sentry_sdk.configure_scope() as scope:
        if (transaction := scope.transaction) is None:
            return ""
        values = [
            f"application='{metadata.__package__}'",
            f"framework='{metadata.__version__}'",
            f"route='{quote(transaction.name)}'",
            f"traceparent='{transaction.trace_id}'",
            f"tracestate='{scope.span.span_id}'",
        ]
        try:
            values.append(f"controller='{scope._tags['account']}'")
        except KeyError:
            pass
        values.append(
            f"action='{';'.join(k for k, v in scope._tags.items() if isinstance(v, bool))}'")
    return " /*" + ",".join(sorted(values)) + "*/"


async def _asyncpg_execute(self,
                           query: str,
                           args,
                           limit,
                           timeout,
                           **kwargs):
    description = query = query.strip()
    if query.startswith("/*"):
        log_sql_probe = query[query.find("*/", 2, 1024) + 3:]
    else:
        log_sql_probe = query
    if _log_sql_re.match(log_sql_probe) and not athenian.api.is_testing:
        from athenian.api.tracing import MAX_SENTRY_STRING_LENGTH
        if len(description) < MAX_SENTRY_STRING_LENGTH and args:
            description += "\n\n" + ", ".join(str(arg) for arg in args)
        if len(description) >= MAX_SENTRY_STRING_LENGTH:
            transaction = sentry_sdk.Hub.current.scope.transaction
            if transaction is not None and transaction.sampled:
                query_id = log_multipart(_sql_log, pickle.dumps((query, args)))
                brief = _sql_str_re.sub("", query)
                description = "%s\n%s" % (query_id, brief[:MAX_SENTRY_STRING_LENGTH])
    with sentry_sdk.start_span(op="sql", description=description) as span:
        if not athenian.api.is_testing:
            query += _generate_tags()
        result = await self._execute_original(query, args, limit, timeout, **kwargs)
        try:
            span.description = "=> %d\n%s" % (len(result[0]), span.description)
        except TypeError:
            pass
        return result


async def _asyncpg_executemany(self, query, args, timeout, **kwargs):
    if timeout is None:
        timeout = float(10 * 60)  # twice as much the default
    with sentry_sdk.start_span(op="sql", description="<= %d\n%s" % (len(args), query)):
        return await self._executemany_original(query, args, timeout, **kwargs)


asyncpg.Connection._execute_original = asyncpg.Connection._Connection__execute
asyncpg.Connection._Connection__execute = _asyncpg_execute
asyncpg.Connection._executemany_original = asyncpg.Connection._executemany
asyncpg.Connection._executemany = _asyncpg_executemany


class greatest(ReturnTypeFromArgs):  # noqa
    """SQL GREATEST function."""


class least(ReturnTypeFromArgs):  # noqa
    """SQL LEAST function."""


db_retry_intervals = [0, 0.1, 0.5, 1.4, None]


def measure_db_overhead_and_retry(db: Union[morcilla.Database, Database],
                                  db_id: Optional[str] = None,
                                  app: Optional[aiohttp.web.Application] = None,
                                  ) -> Union[morcilla.Database, Database]:
    """
    Instrument Database to measure the time spent inside DB i/o.

    Also retry queries after connectivity errors.
    """
    log = logging.getLogger("%s.measure_db_overhead_and_retry" % metadata.__package__)
    backend_connection = db._backend.connection  # type: Callable[[], ConnectionBackend]

    def wrapped_backend_connection() -> ConnectionBackend:
        connection = backend_connection()
        connection._retry_lock = asyncio.Lock()
        direct_acquire = connection.acquire

        def measure_method_overhead_and_retry(func) -> Callable:
            async def wrapped_measure_method_overhead_and_retry(*args, **kwargs):
                start_time = time.time()
                wait_intervals = []
                try:
                    if _conn_backend_in_transaction(connection.raw_connection):
                        # it is pointless to retry, the transaction has already failed
                        wait_intervals = [None]
                except AssertionError:
                    pass  # Connection is not acquired
                if not wait_intervals:
                    wait_intervals = db_retry_intervals

                async def execute():
                    need_acquire = False
                    for i, wait_time in enumerate(wait_intervals):
                        try:
                            if need_acquire:
                                log.warning("Reconnecting to %s", db.url)
                                await direct_acquire()
                                log.info("Connected to %s", db.url)
                            if func == direct_acquire and need_acquire:
                                return
                            return await func(*args, **kwargs)
                        except (OSError,
                                asyncpg.PostgresConnectionError,
                                asyncpg.OperatorInterventionError,
                                asyncpg.InsufficientResourcesError,
                                sqlite3.OperationalError,
                                asyncio.TimeoutError) as e:
                            if wait_time is None:
                                raise e from None
                            log.warning("[%d] %s: %s", i + 1, type(e).__name__, e)
                            if need_acquire := isinstance(
                                e, (asyncpg.PostgresConnectionError, asyncio.TimeoutError),
                            ):
                                try:
                                    connection.raw_connection is not None  # noqa: B015
                                except AssertionError:
                                    # already disconnected
                                    pass
                                else:
                                    log.warning("Disconnecting from %s", db.url)
                                    try:
                                        await connection.release()
                                    except Exception as e:
                                        log.warning("connection.release() raised %s: %s",
                                                    type(e).__name__, e)
                                    else:
                                        log.info("Disconnected from %s", db.url)
                            await asyncio.sleep(wait_time)
                        finally:
                            if app is not None:
                                elapsed = app["db_elapsed"].get()
                                if elapsed is None:
                                    log.warning("Cannot record the %s overhead", db_id)
                                else:
                                    delta = time.time() - start_time
                                    elapsed[db_id] += delta
                if db.url.dialect == "sqlite":
                    return await execute()
                async with connection._retry_lock:
                    return await execute()

            return wraps(wrapped_measure_method_overhead_and_retry, func)

        connection.acquire = measure_method_overhead_and_retry(connection.acquire)
        connection.fetch_all = measure_method_overhead_and_retry(connection.fetch_all)
        connection.fetch_one = measure_method_overhead_and_retry(connection.fetch_one)
        connection.execute = measure_method_overhead_and_retry(connection.execute)
        connection.execute_many = measure_method_overhead_and_retry(connection.execute_many)
        connection.execute_many_native = \
            measure_method_overhead_and_retry(connection.execute_many_native)

        original_transaction = connection.transaction

        def transaction() -> TransactionBackend:
            t = original_transaction()
            t.start = measure_method_overhead_and_retry(t.start)
            t.commit = measure_method_overhead_and_retry(t.commit)
            t.rollback = measure_method_overhead_and_retry(t.rollback)
            return t

        connection.transaction = transaction
        return connection

    db._backend.connection = wrapped_backend_connection
    return db


def check_schema_versions(metadata_db: str,
                          state_db: str,
                          precomputed_db: str,
                          persistentdata_db: str,
                          log: logging.Logger,
                          ) -> bool:
    """Validate schema versions in parallel threads."""
    passed = True
    logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)

    def check_alembic(name, cs):
        nonlocal passed
        try:
            check_alembic_schema_version(name, cs, log)
            check_collation(cs)
        except DBSchemaMismatchError as e:
            passed = False
            log.error("%s schema version check failed: %s", name, e)
        except Exception:
            passed = False
            log.exception("while checking %s", name)

    def check_metadata(cs):
        nonlocal passed
        try:
            check_mdb_schema_version(cs, log)
            check_collation(cs)
        except DBSchemaMismatchError as e:
            passed = False
            log.error("metadata schema version check failed: %s", e)
        except Exception:
            passed = False
            log.exception("while checking metadata")

    checkers = [threading.Thread(target=check_alembic, args=args)
                for args in (("state", state_db),
                             ("precomputed", precomputed_db),
                             ("persistentdata", persistentdata_db),
                             )]
    checkers.append(threading.Thread(target=check_metadata, args=(metadata_db,)))
    for t in checkers:
        t.start()
    for t in checkers:
        t.join()
    return passed


DatabaseLike = Union[Database, Connection]

# https://stackoverflow.com/questions/49456158/integer-in-python-pandas-becomes-blob-binary-in-sqlite  # noqa
for dtype in (np.uint32, np.int32, np.uint64, np.int64):
    sqlite3.register_adapter(dtype, lambda val: int(val))


def _with_statement_hint(self, text, dialect_name="*"):
    if text:
        self._statement_hints += ((dialect_name, text),)
    return self


CompoundSelect._statement_hints = ()
CompoundSelect.with_statement_hint = _with_statement_hint


@compiles(Select)
@compiles(CompoundSelect)
def _visit_select(element, compiler, **kw):
    """Prepend pg_hint_plan hints."""
    per_dialect = [
        ht
        for (dialect_name, ht) in element._statement_hints
        if dialect_name in ("*", compiler.dialect.name)
    ]
    if per_dialect:
        hints = "/*+\n    %s\n */\n" % "\n    ".join(per_dialect)
        statement_hints = element._statement_hints
        element._statement_hints = ()
    else:
        hints = ""
        statement_hints = ()
    try:
        text = getattr(compiler, f"visit_{element.__visit_name__}")(element, **kw)
    finally:
        element._statement_hints = statement_hints
    if dtype := getattr(element, "dtype", None):
        hints = set_query_dtype(hints, dtype)
    if hints:
        return hints + text
    return text


async def insert_or_ignore(model,
                           values: List[Mapping[str, Any]],
                           caller: str,
                           db: Database) -> None:
    """Insert records to the table corresponding to the `model`. Ignore PK collisions."""
    if db.url.dialect == "postgresql":
        sql = postgres_insert(model).on_conflict_do_nothing()
    elif db.url.dialect == "sqlite":
        sql = insert(model).prefix_with("OR IGNORE")
    else:
        raise AssertionError(f"Unsupported database dialect: {db.url.dialect}")
    with sentry_sdk.start_span(op=f"{caller}/execute_many"):
        if db.url.dialect == "sqlite":
            async with db.connection() as pdb_conn:
                async with pdb_conn.transaction():
                    await pdb_conn.execute_many(sql, values)
        else:
            await db.execute_many(sql, values)


def extract_registered_models(base: Any) -> Mapping[str, Any]:
    """Return the mapping from declarative model names to their classes."""
    return base.registry._class_registry


async def conn_in_transaction(con: Connection) -> bool:
    """Return True if the connection is inside a transaction."""
    async with con.raw_connection() as raw_connection:
        return _conn_backend_in_transaction(raw_connection)


def _conn_backend_in_transaction(raw_connection: Any) -> bool:
    """Return True if the connection backend is inside a transaction."""
    if isinstance(raw_connection, asyncpg.Connection):
        return raw_connection.is_in_transaction()
    elif isinstance(raw_connection, aiosqlite.Connection):
        return raw_connection.in_transaction
    raise AssertionError(f"Unhandled db connection type {type(raw_connection)}")


async def dialect_specific_insert(con: Connection) -> Callable:
    """Return the specific insertion function for the connection's SQL dialect."""
    async with con.raw_connection() as raw_connection:
        if isinstance(raw_connection, aiosqlite.Connection):
            return sqlite_insert
        elif isinstance(raw_connection, asyncpg.Connection):
            return postgres_insert
        raise AssertionError(f"Unhandled morcilla connection type {type(raw_connection)}")


async def is_postgresql(db: DatabaseLike) -> bool:
    """Return the value indicating whether the db/connection points at PostgreSQL."""
    if isinstance(db, Database):
        return db.url.dialect == "postgresql"
    else:
        async with db.raw_connection() as raw_connection:
            return isinstance(raw_connection, asyncpg.Connection)
