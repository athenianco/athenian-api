import asyncio
from contextvars import ContextVar
import logging
import os
import time
from typing import Callable, List, Mapping, Tuple, Union

import aiohttp.web
import databases.core
from databases.interfaces import ConnectionBackend, TransactionBackend
from sqlalchemy.sql import ClauseElement

from athenian.api import metadata
from athenian.api.typing_utils import wraps


profile_queries = os.getenv("PROFILE_QUERIES") in ("1", "true", "yes")


def measure_db_overhead(db: databases.Database,
                        db_id: str,
                        app: aiohttp.web.Application) -> databases.Database:
    """Instrument Database to measure the time spent inside DB i/o."""
    log = logging.getLogger("%s.measure_db_overhead" % metadata.__package__)
    _profile_queries = profile_queries

    def measure_method_overhead_and_retry(func) -> callable:
        async def wrapped_measure_method_overhead_and_retry(*args, **kwargs):
            start_time = time.time()
            attempts = 4
            for i in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except OSError as e:
                    if i == attempts - 1:
                        raise e from None
                    log.error("[%d] %s: %s", i + 1, type(e).__name__, e)
                    await asyncio.sleep((i + 1) * (i + 1) * 0.1)
                finally:
                    elapsed = app["db_elapsed"].get()
                    if elapsed is None:
                        log.warning("Cannot record the %s overhead", db_id)
                    else:
                        delta = time.time() - start_time
                        elapsed[db_id] += delta
                        if _profile_queries:
                            sql = str(args[0]).replace("\n", "\\n").replace("\t", "\\t")
                            print("%f\t%s" % (delta, sql), flush=True)

        return wraps(wrapped_measure_method_overhead_and_retry, func)

    backend_connection = db._backend.connection  # type: Callable[[], ConnectionBackend]

    def wrapped_backend_connection() -> ConnectionBackend:
        connection = backend_connection()
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
    pdb_metrics_logger.info("hits/%s: %d", topic, value)


def set_pdb_misses(pdb: databases.Database, topic: str, value: int) -> None:
    """Assign the `topic` precomputed DB misses to `value`."""
    pdb.metrics["misses"].get()[topic] = value
    pdb_metrics_logger.info("misses/%s: %d", topic, value)


def add_pdb_hits(pdb: databases.Database, topic: str, value: int) -> None:
    """Increase the `topic` precomputed hits by `value`."""
    if value < 0:
        pdb_metrics_logger.error('negative add_pdb_hits("%s", %d)', topic, value)
    pdb.metrics["hits"].get()[topic] += value
    pdb_metrics_logger.info("hits/%s: +%d", topic, value)


def add_pdb_misses(pdb: databases.Database, topic: str, value: int) -> None:
    """Increase the `topic` precomputed misses by `value`."""
    if value < 0:
        pdb_metrics_logger.error('negative add_pdb_misses("%s", %d)', topic, value)
    pdb.metrics["misses"].get()[topic] += value
    pdb_metrics_logger.info("misses/%s: +%d", topic, value)


class ParallelDatabase(databases.Database):
    """Override connection() to ignore the task context and spawn a new Connection every time."""

    def __str__(self):
        """Make Sentry debugging easier."""
        return "ParallelDatabase('%s', options=%s)" % (self.url, self.options)

    def connection(self) -> "databases.core.Connection":
        """Bypass self._connection_context."""
        return databases.core.Connection(self._backend)

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

    async def _connection_execute_many(self,
                                       connection: databases.core.Connection,
                                       query: Union[ClauseElement, str],
                                       values: List[Mapping]) -> None:
        """Leverage executemany() if connected to PostgreSQL."""
        if self.url.dialect not in ("postgres", "postgresql"):
            return await connection.execute_many(query, values)
        sql, args = self._compile(query, values)
        async with connection._query_lock:
            await connection._connection.raw_connection.executemany(sql, args)

    async def execute_many(self,
                           query: Union[ClauseElement, str],
                           values: List[Mapping]) -> None:
        """Re-implement execute_many for better performance."""
        async with self.connection() as connection:
            return await self._connection_execute_many(connection, query, values)
