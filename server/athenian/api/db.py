import asyncio
from contextvars import ContextVar
import json
import logging
import os
import pickle
import re
import sqlite3
import sys
import threading
import time
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union
from urllib.parse import quote
import weakref

import aiohttp.web
import aiosqlite
import asyncpg
import databases.core
from databases.interfaces import ConnectionBackend, TransactionBackend
import numpy as np
import sentry_sdk
from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import hstore, insert as postgres_insert
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.ddl import DDLElement
from sqlalchemy.sql.functions import ReturnTypeFromArgs

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


pdb_metrics_logger = logging.getLogger("%s.pdb" % metadata.__package__)


def set_pdb_hits(pdb: databases.Database, topic: str, value: int) -> None:
    """Assign the `topic` precomputed DB hits to `value`."""
    pdb.metrics["hits"].get()[topic] = value
    pdb_metrics_logger.info("hits/%s: %d", topic, value, stacklevel=2)


def set_pdb_misses(pdb: databases.Database, topic: str, value: int) -> None:
    """Assign the `topic` precomputed DB misses to `value`."""
    pdb.metrics["misses"].get()[topic] = value
    pdb_metrics_logger.info("misses/%s: %d", topic, value, stacklevel=2)


def add_pdb_hits(pdb: databases.Database, topic: str, value: int) -> None:
    """Increase the `topic` precomputed hits by `value`."""
    if value < 0:
        pdb_metrics_logger.error('negative add_pdb_hits("%s", %d)', topic, value)
    pdb.metrics["hits"].get()[topic] += value
    pdb_metrics_logger.info("hits/%s: +%d", topic, value, stacklevel=2)


def add_pdb_misses(pdb: databases.Database, topic: str, value: int) -> None:
    """Increase the `topic` precomputed misses by `value`."""
    if value < 0:
        pdb_metrics_logger.error('negative add_pdb_misses("%s", %d)', topic, value)
    pdb.metrics["misses"].get()[topic] += value
    pdb_metrics_logger.info("misses/%s: +%d", topic, value, stacklevel=2)


class FastConnection(databases.core.Connection):
    """Connection with a better execute_many()."""

    def __init__(self, backend: databases.core.DatabaseBackend) -> None:
        """Initialize a new instance of FastConnection."""
        super().__init__(backend)
        self._locked = False  # a poor man's recursive lock for SQLite

    async def fetch_all(self,
                        query: Union[ClauseElement, str],
                        values: dict = None) -> List[Mapping]:
        """Avoid re-wrapping the returned rows in PostgreSQL."""
        if isinstance(self.raw_connection, asyncpg.Connection):
            sql, args = self._compile(self._build_query(query, values), None)
            async with self._query_lock:
                return await self.raw_connection.fetch(sql, *args)
        return await super().fetch_all(query=query, values=values)

    async def fetch_one(self,
                        query: Union[ClauseElement, str],
                        values: dict = None,
                        ) -> Optional[Mapping]:
        """Avoid re-wrapping the returned row in PostgreSQL."""
        if isinstance(self.raw_connection, asyncpg.Connection):
            sql, args = self._compile(self._build_query(query, values), None)
            async with self._query_lock:
                return await self.raw_connection.fetchrow(sql, *args)
        return await super().fetch_one(query=query, values=values)

    async def fetch_val(self,
                        query: Union[ClauseElement, str],
                        values: dict = None,
                        column: Any = 0,
                        ) -> Any:
        """Avoid re-wrapping the returned value in PostgreSQL."""
        if isinstance(self.raw_connection, asyncpg.Connection):
            sql, args = self._compile(self._build_query(query, values), None)
            async with self._query_lock:
                return await self.raw_connection.fetchval(sql, *args, column=column)
        return await super().fetch_val(query=query, column=column)

    async def execute_many(self,
                           query: Union[ClauseElement, str],
                           values: List[Mapping]) -> None:
        """Leverage executemany() if connected to PostgreSQL for better performance."""
        if not isinstance(self.raw_connection, asyncpg.Connection):
            assert self._locked  # sqlite requires wrapping every execute_many in a transaction
            return await super().execute_many(query, values)
        async with self._query_lock:
            return await self.raw_connection.executemany(*self._compile(query, values))

    async def execute(self,
                      query: Union[ClauseElement, str],
                      values: dict = None,
                      ) -> Any:
        """Invoke the parent's execute() with a write serialization lock on SQLite."""  # noqa
        if not isinstance(self.raw_connection, asyncpg.Connection):
            if not self._locked:
                async with self._serialization_lock:
                    self._locked = True
                    try:
                        return await super().execute(query, values)
                    finally:
                        self._locked = False
            else:
                return await super().execute(query, values)
        built_query = self._build_query(query, values)
        query, args = self._compile(built_query, None)
        async with self._query_lock:
            return await self.raw_connection.fetchval(query, *args)

    def transaction(self, *, force_rollback: bool = False, **kwargs: Any,
                    ) -> databases.core.Transaction:
        """Serialize transactions if running on SQLite."""
        transaction = super().transaction(force_rollback=force_rollback, **kwargs)
        if isinstance(self.raw_connection, asyncpg.Connection):
            return transaction

        original_start = transaction.start
        original_rollback = transaction.rollback
        original_commit = transaction.commit

        async def start_transaction() -> databases.core.Transaction:
            assert not self._locked
            await self._serialization_lock.acquire()
            self._locked = True
            return await original_start()

        async def rollback_transaction() -> None:
            assert self._locked
            self._locked = False
            self._serialization_lock.release()
            return await original_rollback()

        async def commit_transaction() -> None:
            assert self._locked
            self._locked = False
            self._serialization_lock.release()
            return await original_commit()

        transaction.start = start_transaction
        transaction.rollback = rollback_transaction
        transaction.commit = commit_transaction
        return transaction

    def _compile(self,
                 query: ClauseElement,
                 values: Optional[List[Mapping]],
                 ) -> Tuple[str, List[list]]:
        compiled = query.compile(dialect=self._backend._dialect)
        if not isinstance(query, DDLElement):
            compiled_params = sorted(compiled.params.items())
            sql_mapping = {
                key: "$" + str(i) for i, (key, _) in enumerate(compiled_params, start=1)
            }
            compiled_query = compiled.string % sql_mapping

            processors = compiled._bind_processors
            if isinstance(self.raw_connection, asyncpg.Connection):
                # we should not process HSTORE and JSON, asyncpg will do it for us
                removed = [key for key, val in processors.items()
                           if val.__qualname__.startswith("HSTORE") or
                           val.__qualname__.startswith("JSON")]
                for key in removed:
                    del processors[key]
            args = []
            if values is not None:
                param_mapping = {key: i for i, (key, _) in enumerate(compiled_params)}
                for dikt in values:
                    series = [None] * len(compiled_params)
                    args.append(series)
                    for key, val in dikt.items():
                        try:
                            val = processors[key](val)
                        except KeyError:
                            pass
                        series[param_mapping[key]] = val
            else:
                for key, val in compiled_params:
                    try:
                        val = processors[key](val)
                    except KeyError:
                        pass
                    args.append(val)
        else:
            compiled_query = compiled.string
            args = []
        return compiled_query, args


class ParallelDatabase(databases.Database):
    """
    Override connection() to ignore the task context and spawn a new Connection every time.

    Tweak the behavior on per-dialect basis.
    """

    _serialization_lock = None
    # please report your naming disgust to the authors of asyncpg
    _introspect_types_cache = {}
    _introspect_type_cache = {}

    def __init__(
        self,
        url: Union[str, databases.DatabaseURL],
        **options: Any,
    ):
        """
        Database constructor with blackjack and whores.

        If running on Postgres, enable parsing HSTORE columns.
        If running on SQLite, initialize the shared write serialization lock.
        """
        url = databases.DatabaseURL(url)
        if url.dialect not in ("postgresql", "sqlite"):
            raise ValueError("Dialect %s is not supported." % url.dialect)
        if url.dialect == "postgresql":
            options["init"] = self._register_codecs
            options["statement_cache_size"] = 0
            self._ignore_hstore = False
            self._introspect_types_cache_db = {}
            self._introspect_type_cache_db = {}
        super().__init__(url, force_rollback=options.pop("force_rollback", False), **options)
        if url.dialect == "sqlite":
            self._serialization_lock = asyncio.Lock()

    def __repr__(self) -> str:
        """Make Sentry debugging easier."""
        return "ParallelDatabase('%s', **%s)" % (self.url, self.options)

    def __str__(self) -> str:
        """Make Sentry debugging easier."""
        return repr(self)

    def connection(self) -> Union[FastConnection, databases.core.Connection]:
        """Bypass self._connection_context."""
        if self.url.database == "":
            # SQLite in-memory
            return super().connection()
        connection = FastConnection(self._backend)
        connection._serialization_lock = self._serialization_lock
        return connection

    def _on_connection_close(self, conn: asyncpg.Connection):
        try:
            del self._introspect_types_cache[weakref.ref(conn)]
            del self._introspect_type_cache[weakref.ref(conn)]
        except TypeError:
            # conn is a PoolConnectionProxy and the DB is dying
            log = logging.getLogger(
                f"{metadata.__package__}.ParallelDatabase._on_connection_close")
            log.warning("could not dispose type introspection caches for connection %s", conn)

    async def _register_codecs(self, conn: asyncpg.Connection) -> None:
        # we have to maintain separate caches for each database because OID-s appear different
        self._introspect_types_cache[weakref.ref(conn)] = self._introspect_types_cache_db
        self._introspect_type_cache[weakref.ref(conn)] = self._introspect_type_cache_db
        conn.add_termination_listener(self._on_connection_close)
        await conn.set_type_codec(
            "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
        if self._ignore_hstore:
            return
        try:
            await conn.set_builtin_type_codec("hstore", codec_name="pg_contrib.hstore")
        except ValueError:
            # no HSTORE is registered
            self._ignore_hstore = True
            databases.core.logger.warning("no HSTORE is registered in %s", self.url)


_sql_log = logging.getLogger("%s.sql" % metadata.__package__)
_testing = "pytest" in sys.modules or os.getenv("SENTRY_ENV", "development") == "development"
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
    if _log_sql_re.match(query) and not _testing:
        from athenian.api.tracing import MAX_SENTRY_STRING_LENGTH
        if len(description) <= MAX_SENTRY_STRING_LENGTH and args:
            description += " | " + str(args)
        if len(description) > MAX_SENTRY_STRING_LENGTH:
            transaction = sentry_sdk.Hub.current.scope.transaction
            if transaction is not None and transaction.sampled:
                query_id = log_multipart(_sql_log, pickle.dumps((query, args)))
                brief = _sql_str_re.sub("", query)
                description = "%s\n%s" % (query_id, brief[:MAX_SENTRY_STRING_LENGTH])
    with sentry_sdk.start_span(op="sql", description=description) as span:
        if not _testing:
            query += _generate_tags()
        result = await self._execute_original(query, args, limit, timeout, **kwargs)
        try:
            span.description = "=> %d\n%s" % (len(result[0]), span.description)
        except TypeError:
            pass
        return result


async def _asyncpg_executemany(self, query, args, timeout, **kwargs):
    with sentry_sdk.start_span(op="sql", description="<= %d\n%s" % (len(args), query)):
        return await self._executemany_original(query, args, timeout, **kwargs)


class _FakeStatement:
    name = ""


async def _introspect_types_cached(self, typeoids, timeout):
    introspect_types_cache = ParallelDatabase._introspect_types_cache[weakref.ref(self)]
    if missing := [oid for oid in typeoids if oid not in introspect_types_cache]:
        rows, stmt = await self._introspect_types_original(missing, timeout)
        assert stmt.name == ""
        for row in rows:
            introspect_types_cache[row["oid"]] = row
    return [introspect_types_cache[oid] for oid in typeoids], _FakeStatement


async def _introspect_type_cached(self, typename, schema):
    introspect_type_cache = ParallelDatabase._introspect_type_cache[weakref.ref(self)]
    try:
        return introspect_type_cache[typename]
    except KeyError:
        introspect_type_cache[typename] = r = await self._introspect_type_original(
            typename, schema)
        return r


asyncpg.Connection._introspect_types_original = asyncpg.Connection._introspect_types
asyncpg.Connection._introspect_types = _introspect_types_cached
asyncpg.Connection._introspect_type_original = asyncpg.Connection._introspect_type
asyncpg.Connection._introspect_type = _introspect_type_cached
asyncpg.Connection._execute_original = asyncpg.Connection._Connection__execute
asyncpg.Connection._Connection__execute = _asyncpg_execute
asyncpg.Connection._executemany_original = asyncpg.Connection._executemany
asyncpg.Connection._executemany = _asyncpg_executemany


hstore = sys.modules[hstore.__module__]
_original_parse_hstore = hstore._parse_hstore


def _universal_parse_hstore(hstore_str):
    if isinstance(hstore_str, dict):
        return hstore_str
    return _original_parse_hstore(hstore_str)


hstore._parse_hstore = _universal_parse_hstore


class greatest(ReturnTypeFromArgs):  # noqa
    """SQL GREATEST function."""


class least(ReturnTypeFromArgs):  # noqa
    """SQL LEAST function."""


db_retry_intervals = [0, 0.1, 0.5, 1.4, None]


def measure_db_overhead_and_retry(db: Union[databases.Database, ParallelDatabase],
                                  db_id: Optional[str] = None,
                                  app: Optional[aiohttp.web.Application] = None,
                                  ) -> Union[databases.Database, ParallelDatabase]:
    """
    Instrument Database to measure the time spent inside DB i/o.

    Also retry queries after connectivity errors.
    """
    log = logging.getLogger("%s.measure_db_overhead_and_retry" % metadata.__package__)
    backend_connection = db._backend.connection  # type: Callable[[], ConnectionBackend]

    def wrapped_backend_connection() -> ConnectionBackend:
        connection = backend_connection()
        connection._retry_lock = asyncio.Lock()

        def measure_method_overhead_and_retry(func) -> callable:
            async def wrapped_measure_method_overhead_and_retry(*args, **kwargs):
                start_time = time.time()
                wait_intervals = []
                try:
                    raw_connection = connection.raw_connection
                    if (
                        (isinstance(raw_connection, asyncpg.Connection) and
                         raw_connection.is_in_transaction())
                        or
                        (isinstance(raw_connection, aiosqlite.Connection) and
                         raw_connection.in_transaction)
                    ):
                        # it is pointless to retry, the transaction is already considered as failed
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
                                await connection.acquire()
                            return await func(*args, **kwargs)
                        except (OSError,
                                asyncpg.PostgresConnectionError,
                                asyncpg.OperatorInterventionError,
                                asyncpg.InsufficientResourcesError,
                                sqlite3.OperationalError) as e:
                            if wait_time is None:
                                raise e from None
                            log.warning("[%d] %s: %s", i + 1, type(e).__name__, e)
                            if need_acquire := isinstance(e, asyncpg.PostgresConnectionError):
                                try:
                                    await connection.release()
                                except Exception as e:
                                    log.warning("connection.release() raised %s: %s",
                                                type(e).__name__, e)
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


DatabaseLike = Union[ParallelDatabase, FastConnection]

# https://stackoverflow.com/questions/49456158/integer-in-python-pandas-becomes-blob-binary-in-sqlite  # noqa
for dtype in (np.uint32, np.int32, np.uint64, np.int64):
    sqlite3.register_adapter(dtype, lambda val: int(val))


async def insert_or_ignore(model,
                           values: List[Mapping[str, Any]],
                           caller: str,
                           db: ParallelDatabase) -> None:
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
