import asyncio
from contextvars import ContextVar
import logging
import os
import pickle
import re
import sys
import time
from typing import Callable, List, Mapping, Tuple, Union

import aiohttp.web
import aiosqlite
import asyncpg
import databases.core
from databases.interfaces import ConnectionBackend, TransactionBackend
import sentry_sdk
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.functions import ReturnTypeFromArgs

from athenian.api import metadata
from athenian.api.slogging import log_multipart
from athenian.api.tracing import MAX_SENTRY_STRING_LENGTH
from athenian.api.typing_utils import wraps


def measure_db_overhead_and_retry(db: databases.Database,
                                  db_id: str,
                                  app: aiohttp.web.Application) -> databases.Database:
    """
    Instrument Database to measure the time spent inside DB i/o.

    Also retry queries after connectivity errors.
    """
    log = logging.getLogger("%s.measure_db_overhead_and_retry" % metadata.__package__)
    backend_connection = db._backend.connection  # type: Callable[[], ConnectionBackend]

    def wrapped_backend_connection() -> ConnectionBackend:
        connection = backend_connection()

        def measure_method_overhead_and_retry(func) -> callable:
            async def wrapped_measure_method_overhead_and_retry(*args, **kwargs):
                start_time = time.time()
                if (
                    (isinstance(connection, asyncpg.Connection) and connection.is_in_transaction())
                    or
                    (isinstance(connection, aiosqlite.Connection) and connection.in_transaction)
                ):
                    wait_intervals = [None]
                else:
                    wait_intervals = [0.1, 0.5, 1.4, None]
                for i, wait_time in enumerate(wait_intervals):
                    try:
                        return await func(*args, **kwargs)
                    except OSError as e:
                        if wait_time is None:
                            raise e from None
                        log.warning("[%d] %s: %s", i + 1, type(e).__name__, e)
                        await asyncio.sleep(wait_time)
                    finally:
                        elapsed = app["db_elapsed"].get()
                        if elapsed is None:
                            log.warning("Cannot record the %s overhead", db_id)
                        else:
                            delta = time.time() - start_time
                            elapsed[db_id] += delta

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


class ExecuteManyConnection(databases.core.Connection):
    """Connection with a better execute_many()."""

    async def execute_many(self,
                           query: Union[ClauseElement, str],
                           values: List[Mapping]) -> None:
        """Leverage executemany() if connected to PostgreSQL for better performance."""
        if not isinstance(self.raw_connection, asyncpg.Connection):
            return await super().execute_many(query, values)
        sql, args = self._compile(query, values)
        async with self._query_lock:
            await self.raw_connection.executemany(sql, args)

    def _compile(self, query: ClauseElement, values: List[Mapping]) -> Tuple[str, List[list]]:
        compiled = query.compile(dialect=self._backend._dialect)
        compiled_params = sorted(compiled.params.items())

        sql_mapping = {}
        param_mapping = {}
        for i, (key, _) in enumerate(compiled_params):
            sql_mapping[key] = "$" + str(i + 1)
            param_mapping[key] = i
        compiled_query = compiled.string % sql_mapping

        processors = compiled._bind_processors
        args = []
        for dikt in values:
            series = [None] * len(compiled_params)
            args.append(series)
            for key, val in dikt.items():
                series[param_mapping[key]] = processors[key](val) if key in processors else val

        return compiled_query, args


class ParallelDatabase(databases.Database):
    """Override connection() to ignore the task context and spawn a new Connection every time."""

    def __str__(self):
        """Make Sentry debugging easier."""
        return "ParallelDatabase('%s', options=%s)" % (self.url, self.options)

    def connection(self) -> "databases.core.Connection":
        """Bypass self._connection_context."""
        return ExecuteManyConnection(self._backend)


_sql_log = logging.getLogger("%s.sql" % metadata.__package__)
_testing = "pytest" in sys.modules or os.getenv("SENTRY_ENV", "development") == "development"
_sql_str_re = re.compile(r"'[^']+'(, )?")
_log_sql_re = re.compile(r"SELECT|\(SELECT|WITH RECURSIVE")


async def _asyncpg_execute(self, query: str, args, limit, timeout, return_status=False):
    description = query = query.strip()
    if _log_sql_re.match(query) and not _testing:
        if len(description) <= MAX_SENTRY_STRING_LENGTH and args:
            description += " | " + str(args)
        if len(description) > MAX_SENTRY_STRING_LENGTH:
            transaction = sentry_sdk.Hub.current.scope.transaction
            if transaction is not None and transaction.sampled:
                query_id = log_multipart(_sql_log, pickle.dumps((query, args)))
                brief = _sql_str_re.sub("", query)
                description = "%s\n%s" % (query_id, brief[:MAX_SENTRY_STRING_LENGTH])
    with sentry_sdk.start_span(op="sql", description=description) as span:
        result = await self._execute_original(query, args, limit, timeout, return_status)
        try:
            span.description = "=> %d\n%s" % (len(result[0]), span.description)
        except TypeError:
            pass
        return result


async def _asyncpg_executemany(self, query, args, timeout):
    with sentry_sdk.start_span(op="sql", description="<= %d\n%s" % (len(args), query)):
        return await self._executemany_original(query, args, timeout)


asyncpg.Connection._execute_original = asyncpg.Connection._Connection__execute
asyncpg.Connection._Connection__execute = _asyncpg_execute
asyncpg.Connection._executemany_original = asyncpg.Connection._executemany
asyncpg.Connection._executemany = _asyncpg_executemany


class greatest(ReturnTypeFromArgs):  # noqa
    """SQL GREATEST function."""


class least(ReturnTypeFromArgs):  # noqa
    """SQL LEAST function."""
